//
//  PipelineCoordinator.swift
//  SightLine
//

import SwiftUI
import AVFoundation
import Combine
import os
import UIKit

private let logger = Logger(subsystem: "com.sightline.app", category: "PipelineCoordinator")

enum DepthModelState {
    case loading
    case ready
    case unavailable
}

enum PermissionPromptReason {
    case microphoneRequired
    case cameraRequired
}

@MainActor
final class PipelineCoordinator: ObservableObject {
    let webSocketManager = WebSocketManager()
    let audioCapture = AudioCaptureManager()
    let audioPlayback = AudioPlaybackManager()
    let cameraManager = CameraManager()
    let frameSelector = FrameSelector()
    let sensorManager = SensorManager()
    let telemetryAggregator = TelemetryAggregator()
    let mediaPermissionGate = MediaPermissionGate()
    let debugModel = DebugOverlayModel()
    let devConsoleModel = DeveloperConsoleModel()
    let thinkingSoundManager = ThinkingSoundManager()

    @Published var transcript: String = ""
    @Published var isActive = false
    @Published var currentLOD: Int = 2
    @Published var connectionStatus: String = "Connecting..."
    @Published var isSafeMode = false
    @Published var whenIdleToolQueue: [String] = []
    @Published var showDevConsole = false
    @Published var isMuted = false
    @Published var isCameraActive = false
    @Published var isPaused = false
    @Published var userAudioEnergy: CGFloat = 0
    @Published var toastMessage: String?
    @Published var showPermissionPrompt = false
    @Published var permissionPromptReason: PermissionPromptReason?
    @Published var depthModelState: DepthModelState = .loading
    @Published var pendingCameraActivation = false
    @Published var sessionResumptionHandle = UserDefaults.standard.string(
        forKey: SightLineConfig.sessionResumptionHandleDefaultsKey
    ) ?? ""

    var lastAgentTranscript = ""
    var sessionReadyTimeoutTask: Task<Void, Never>?
    var hasReceivedSessionReady = false

    private let localSynthesizer = AVSpeechSynthesizer()
    private var cancellables = Set<AnyCancellable>()

    lazy var messageRouter = MessageRouter(coordinator: self)

    init() {
        bindNestedObjects()
    }

    private func bindNestedObjects() {
        let publishers: [AnyPublisher<Void, Never>] = [
            webSocketManager.objectWillChange.eraseToAnyPublisher(),
            audioCapture.objectWillChange.eraseToAnyPublisher(),
            audioPlayback.objectWillChange.eraseToAnyPublisher(),
            cameraManager.objectWillChange.eraseToAnyPublisher(),
            frameSelector.objectWillChange.eraseToAnyPublisher(),
            sensorManager.objectWillChange.eraseToAnyPublisher(),
            telemetryAggregator.objectWillChange.eraseToAnyPublisher(),
            debugModel.objectWillChange.eraseToAnyPublisher(),
            devConsoleModel.objectWillChange.eraseToAnyPublisher(),
        ]

        publishers.forEach { publisher in
            publisher
                .sink { [weak self] _ in
                    self?.objectWillChange.send()
                }
                .store(in: &cancellables)
        }
    }

    func handleSingleTap() {
        isMuted.toggle()
        HapticManager.shared.singleTap()
        if isMuted {
            audioCapture.stopCapture()
        } else if !isPaused {
            audioCapture.startCapture()
        }
        webSocketManager.sendText(UpstreamMessage.muteToggle(muted: isMuted).toJSON())
        UIAccessibility.post(notification: .announcement, argument: isMuted ? "Microphone muted" : "Microphone unmuted")
        devConsoleModel.captureTranscript(text: "Gesture: mute_toggle (muted=\(isMuted))", role: "gesture")
        logger.info("Gesture: mute_toggle (isMuted=\(self.isMuted))")
    }

    func handleDoubleTap() {
        HapticManager.shared.doubleTap()
        audioPlayback.stopImmediately()
        audioPlayback.suppressIncomingAudio(for: 0.8)
        webSocketManager.sendText(UpstreamMessage.gesture(type: "interrupt").toJSON())
        UIAccessibility.post(notification: .announcement, argument: "Speech interrupted")
        devConsoleModel.captureTranscript(text: "Gesture: interrupt", role: "gesture")
        showToast("Speech interrupted")
        logger.info("Gesture: interrupt")
    }

    func handleTripleTap() {
        HapticManager.shared.tripleTap()
        webSocketManager.sendText(UpstreamMessage.gesture(type: "repeat_last").toJSON())
        UIAccessibility.post(notification: .announcement, argument: "Repeating last message")
        devConsoleModel.captureTranscript(text: "Gesture: repeat_last", role: "gesture")
        showToast("Repeating last message")
        logger.info("Gesture: repeat_last")
    }

    func handleLongPress() {
        isPaused.toggle()
        HapticManager.shared.longPress()
        if isPaused {
            audioCapture.stopCapture()
            audioPlayback.stopImmediately()
            if isCameraActive {
                cameraManager.stopCapture()
                isCameraActive = false
            }
        } else {
            if !isMuted {
                audioCapture.startCapture()
            }
            // Camera is NOT auto-resumed — user must explicitly re-enable via swipe.
        }
        webSocketManager.sendText(UpstreamMessage.pause(paused: isPaused).toJSON())
        UIAccessibility.post(notification: .announcement, argument: isPaused ? "Session paused" : "Session resumed")
        devConsoleModel.captureTranscript(text: "Gesture: pause (paused=\(isPaused))", role: "gesture")
        logger.info("Gesture: pause (paused=\(self.isPaused))")
    }

    func handleSwipe(translation: CGSize) {
        if abs(translation.height) > abs(translation.width) {
            // Vertical swipe: LOD upgrade/downgrade
            HapticManager.shared.swipe()
            if translation.height < 0 {
                webSocketManager.sendText(UpstreamMessage.gesture(type: "lod_up").toJSON())
                UIAccessibility.post(notification: .announcement, argument: "Detail level increasing")
                devConsoleModel.captureTranscript(text: "Gesture: lod_up", role: "gesture")
                logger.info("Gesture: lod_up")
            } else {
                webSocketManager.sendText(UpstreamMessage.gesture(type: "lod_down").toJSON())
                UIAccessibility.post(notification: .announcement, argument: "Detail level decreasing")
                devConsoleModel.captureTranscript(text: "Gesture: lod_down", role: "gesture")
                logger.info("Gesture: lod_down")
            }
        } else {
            // Horizontal swipe: toggle camera on/off
            devConsoleModel.captureTranscript(text: "Gesture: camera_toggle", role: "gesture")
            toggleCamera()
        }
    }

    func currentPreferredLanguage() -> String {
        UserDefaults.standard.string(forKey: SightLineConfig.preferredLanguageDefaultsKey) ?? "en-US"
    }

    func localizedText(
        english: String,
        chinese: String,
        spanish: String,
        japanese: String
    ) -> String {
        let language = currentPreferredLanguage().lowercased()
        if language.hasPrefix("zh") { return chinese }
        if language.hasPrefix("es") { return spanish }
        if language.hasPrefix("ja") { return japanese }
        return english
    }

    func speakLocalStatus(_ message: String) {
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: currentPreferredLanguage())
            ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        localSynthesizer.speak(utterance)
    }

    func cameraPreparingMessage() -> String {
        localizedText(
            english: "Preparing camera analysis. Please wait.",
            chinese: "正在准备相机分析，请稍候。",
            spanish: "Preparando el analisis de la camara. Espera un momento.",
            japanese: "カメラ解析を準備しています。少しお待ちください。"
        )
    }

    func cameraDepthUnavailableMessage() -> String {
        localizedText(
            english: "Depth alerts are unavailable. Camera descriptions are still on.",
            chinese: "深度提醒暂时不可用，但相机描述仍然可用。",
            spanish: "Las alertas de profundidad no estan disponibles, pero la camara sigue describiendo la escena.",
            japanese: "深度アラートは利用できませんが、カメラの説明は引き続き使えます。"
        )
    }

    func sessionReadyTimedOutMessage() -> String {
        localizedText(
            english: "Session initialization timed out. Reconnecting now.",
            chinese: "会话初始化超时，正在重新连接。",
            spanish: "La inicializacion de la sesion agoto el tiempo. Reconectando ahora.",
            japanese: "セッションの初期化がタイムアウトしました。再接続しています。"
        )
    }

    func connectionLostMessage() -> String {
        localizedText(
            english: "Connection lost. Safe mode active.",
            chinese: "连接已断开，已进入安全模式。",
            spanish: "Se perdio la conexion. El modo seguro esta activo.",
            japanese: "接続が切れました。セーフモードを有効にしました。"
        )
    }

    func connectionRestoredMessage() -> String {
        localizedText(
            english: "Connection restored.",
            chinese: "连接已恢复。",
            spanish: "Conexion restablecida.",
            japanese: "接続が回復しました。"
        )
    }

    func capabilityDegradedMessage(for capability: String) -> String {
        switch capability.lowercased() {
        case "vision":
            return localizedText(
                english: "Visual analysis is temporarily unavailable.",
                chinese: "视觉分析暂时不可用。",
                spanish: "El analisis visual no esta disponible temporalmente.",
                japanese: "視覚解析は一時的に利用できません。"
            )
        case "face":
            return localizedText(
                english: "Face recognition is temporarily unavailable.",
                chinese: "人脸识别暂时不可用。",
                spanish: "El reconocimiento facial no esta disponible temporalmente.",
                japanese: "顔認識は一時的に利用できません。"
            )
        case "ocr":
            return localizedText(
                english: "Text reading is temporarily unavailable.",
                chinese: "文字识别暂时不可用。",
                spanish: "La lectura de texto no esta disponible temporalmente.",
                japanese: "文字読み取りは一時的に利用できません。"
            )
        default:
            return localizedText(
                english: "A feature is temporarily unavailable.",
                chinese: "有一项功能暂时不可用。",
                spanish: "Una funcion no esta disponible temporalmente.",
                japanese: "一部の機能が一時的に利用できません。"
            )
        }
    }

    func scheduleSessionReadyTimeout() {
        sessionReadyTimeoutTask?.cancel()
        hasReceivedSessionReady = false
        sessionReadyTimeoutTask = Task {
            try? await Task.sleep(nanoseconds: 30_000_000_000)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard isActive, !hasReceivedSessionReady else { return }
                let message = sessionReadyTimedOutMessage()
                connectionStatus = message
                transcript = message
                audioCapture.stopCapture()
                cameraManager.stopCapture()
                audioPlayback.stopImmediately()
                isCameraActive = false
                showToast(message)
                speakLocalStatus(message)
                webSocketManager.reconnect(afterMs: 1000)
            }
        }
    }

    func cancelSessionReadyTimeout() {
        sessionReadyTimeoutTask?.cancel()
        sessionReadyTimeoutTask = nil
    }

    func activateCamera(announceDepthUnavailable: Bool) {
        guard isActive, !isPaused else {
            pendingCameraActivation = false
            return
        }
        cameraManager.startCapture()
        isCameraActive = true
        pendingCameraActivation = false
        connectionStatus = isSafeMode ? "Safe Mode - Reconnecting..." : "Connected"
        HapticManager.shared.cameraOn()
        webSocketManager.sendText(UpstreamMessage.cameraToggle(active: true).toJSON())
        if announceDepthUnavailable {
            let message = cameraDepthUnavailableMessage()
            transcript = message
            showToast(message)
            UIAccessibility.post(notification: .announcement, argument: message)
        } else {
            UIAccessibility.post(notification: .announcement, argument: "Camera on")
        }
        logger.info("Camera activated by user")
    }

    func showToast(_ message: String) {
        toastMessage = message
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            guard let self else { return }
            if self.toastMessage == message {
                self.toastMessage = nil
            }
        }
    }

    var shouldCaptureDevEvents: Bool {
        #if DEBUG
        return showDevConsole
        #else
        return false
        #endif
    }

    func openAppSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    func presentPermissionPrompt(
        reason: PermissionPromptReason,
        status: String,
        transcript: String? = nil,
        announcement: String
    ) {
        permissionPromptReason = reason
        showPermissionPrompt = true
        connectionStatus = status
        if let transcript {
            self.transcript = transcript
        }
        UIAccessibility.post(notification: .announcement, argument: announcement)
    }

    func dismissPermissionPrompt() {
        showPermissionPrompt = false
        permissionPromptReason = nil
    }

    /// Recheck permissions when returning from Settings. Dismiss prompt if granted.
    /// If WebSocket was blocked due to missing mic permission, connect now.
    func recheckPermissionsAfterSettings() {
        Task {
            let micGranted = await mediaPermissionGate.ensureMicPermission()
            let cameraDenied = mediaPermissionGate.isCameraPermissionDenied()
            await MainActor.run {
                guard micGranted else {
                    presentPermissionPrompt(
                        reason: .microphoneRequired,
                        status: "Microphone permission required",
                        transcript: "Please enable microphone permission.",
                        announcement: "Microphone permission is required. Use Open Settings button to grant access."
                    )
                    return
                }

                switch permissionPromptReason {
                case .cameraRequired:
                    guard cameraDenied else {
                        dismissPermissionPrompt()
                        transcript = "Camera permission restored. Swipe sideways to turn on camera."
                        showToast("Camera permission restored")
                        UIAccessibility.post(
                            notification: .announcement,
                            argument: "Camera permission restored. Swipe sideways to turn on camera."
                        )
                        return
                    }
                    presentPermissionPrompt(
                        reason: .cameraRequired,
                        status: "Camera permission required",
                        transcript: "Camera permission required. Enable in Settings.",
                        announcement: "Camera permission required. Use Open Settings button to grant access."
                    )

                case .microphoneRequired, .none:
                    dismissPermissionPrompt()
                    connectionStatus = webSocketManager.isConnected ? "Connected" : "Permissions granted"
                    UIAccessibility.post(
                        notification: .announcement,
                        argument: "Microphone permission granted. Starting audio."
                    )
                    if !webSocketManager.isConnected {
                        if let url = SightLineConfig.wsURL(
                            userId: SightLineConfig.defaultUserId,
                            sessionId: SightLineConfig.defaultSessionId,
                            resumeHandle: sessionResumptionHandle.isEmpty ? nil : sessionResumptionHandle
                        ) {
                            webSocketManager.connect(url: url)
                        }
                    } else {
                        startAudioCapture()
                    }
                }
            }
        }
    }

    func setupPipeline() {
        // 1. Build WebSocket URL and wire callbacks (configured first, connected after permissions)
        guard let url = SightLineConfig.wsURL(
            userId: SightLineConfig.defaultUserId,
            sessionId: SightLineConfig.defaultSessionId,
            resumeHandle: sessionResumptionHandle.isEmpty ? nil : sessionResumptionHandle
        ) else {
            logger.error("Invalid WebSocket URL configuration")
            connectionStatus = "Configuration error"
            return
        }

        webSocketManager.onTextSent = { [weak self] text in
            DispatchQueue.main.async {
                guard let self else { return }
                if self.shouldCaptureDevEvents {
                    self.devConsoleModel.captureNetworkMessage(direction: "UP", payload: text)
                }
            }
        }

        webSocketManager.onAudioReceived = { [weak self] data in
            self?.thinkingSoundManager.stopImmediately()
            self?.audioCapture.lastModelAudioReceivedAt = CFAbsoluteTimeGetCurrent()
            self?.audioPlayback.playAudioData(data)
        }

        webSocketManager.onPrioritizedAudioReceived = { [weak self] data, priority in
            self?.thinkingSoundManager.stopImmediately()
            self?.audioCapture.lastModelAudioReceivedAt = CFAbsoluteTimeGetCurrent()
            self?.audioPlayback.playAudioData(data, priority: priority)
        }

        audioCapture.isModelAudioPlaying = { [weak self] in
            self?.audioPlayback.isPlaying ?? false
        }

        audioCapture.onVoiceBargeIn = { [weak self] in
            guard let self else { return }
            DispatchQueue.main.async {
                self.audioPlayback.stopImmediately()
                // Short suppression to drop in-flight chunks; the server
                // will stop forwarding once it processes client_barge_in.
                self.audioPlayback.suppressIncomingAudio(for: 0.5)
            }
            // With NO_INTERRUPTION, Gemini's server-side VAD won't interrupt
            // the model.  Instead, notify the server so it stops forwarding
            // audio until the current turn completes.
            self.webSocketManager.sendText(UpstreamMessage.clientBargeIn.toJSON())
        }

        // Manual activity_start/activity_end signals removed:
        // Server-side VAD is re-enabled with conservative sensitivity.
        // Client-side RMS gating acts as AEC; explicit signals are unnecessary
        // and conflict with automatic activity detection.

        webSocketManager.onTextReceived = { [weak self] text in
            guard let self else { return }
            DispatchQueue.main.async {
                if self.shouldCaptureDevEvents {
                    self.devConsoleModel.captureNetworkMessage(direction: "DOWN", payload: text)
                }
            }
            if let msg = DownstreamMessage.parse(text: text) {
                self.messageRouter.handleDownstreamMessage(msg)
            }
        }

        webSocketManager.onConnectionStateChanged = { [weak self] connected in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isActive = connected
                self.debugModel.isConnected = connected
                if connected {
                    self.connectionStatus = "Connected"
                    self.scheduleSessionReadyTimeout()
                } else {
                    self.cancelSessionReadyTimeout()
                    self.audioCapture.stopCapture()
                    self.cameraManager.stopCapture()
                    self.audioPlayback.stopImmediately()
                    self.isCameraActive = false
                    self.pendingCameraActivation = false
                }
            }
        }

        // SL-38: Disconnection degradation callbacks
        webSocketManager.onDisconnectionDegraded = { [weak self] in
            DispatchQueue.main.async {
                self?.enterSafeMode()
            }
        }

        webSocketManager.onConnectionRestored = { [weak self] in
            DispatchQueue.main.async {
                self?.exitSafeMode()
            }
        }

        // Await permissions before connecting — mic is required, camera is optional (audio-only mode)
        Task {
            let micGranted = await mediaPermissionGate.ensureMicPermission()
            let cameraDenied = mediaPermissionGate.isCameraPermissionDenied()

            if !micGranted {
                // Microphone is essential — cannot function without it
                await MainActor.run {
                    presentPermissionPrompt(
                        reason: .microphoneRequired,
                        status: "Microphone permission required",
                        announcement: "Microphone permission is required. Use Open Settings button to grant access."
                    )
                }
                logger.error("WebSocket connection blocked: microphone permission denied")
                return
            }

            if cameraDenied {
                logger.info("Camera permission denied at launch; deferring prompt until user requests camera")
            }

            // Connect only after mic permission confirmed
            await MainActor.run {
                dismissPermissionPrompt()
                webSocketManager.connect(url: url)
            }
        }

        // 2. Setup camera with LOD-based frame selector + pixel-diff dedup (SL-75)
        cameraManager.onCameraFailure = { [weak self] reason in
            self?.webSocketManager.sendText(UpstreamMessage.cameraFailure(error: reason).toJSON())
        }
        cameraManager.frameSelector = frameSelector
        cameraManager.onFrameCaptured = { [weak self] jpegData in
            guard let self else { return }
            guard self.frameSelector.isFrameDifferent(jpegData: jpegData) else {
                self.frameSelector.markFrameSkipped()
                logger.debug("Frame skipped (pixel-diff below threshold)")
                return
            }
            // Phase 5: Binary frame optimization — eliminates ~33% Base64 overhead
            let msg = UpstreamMessage.image(data: jpegData, mimeType: "image/jpeg")
            if let binaryFrame = msg.toBinary() {
                self.webSocketManager.sendBinary(binaryFrame)
            } else {
                self.webSocketManager.sendText(msg.toJSON())
            }
            self.frameSelector.markFrameSent()
        }

        // 2b. Setup depth estimation pipeline (CoreML Depth Anything V2)
        depthModelState = .loading
        pendingCameraActivation = false
        let depthEstimator = DepthEstimator()
        cameraManager.depthEstimator = depthEstimator
        cameraManager.onDepthEstimated = { [weak self] summary in
            self?.sensorManager.updateDepth(summary)
        }
        Task(priority: .userInitiated) {
            let available = await MainActor.run {
                depthEstimator.loadModel()
                return depthEstimator.isAvailable
            }

            await MainActor.run {
                self.depthModelState = available ? .ready : .unavailable
                logger.info("Depth model async load complete (available: \(available))")

                guard self.pendingCameraActivation else { return }
                if available {
                    self.connectionStatus = "Connected"
                    self.activateCamera(announceDepthUnavailable: false)
                } else {
                    self.activateCamera(announceDepthUnavailable: true)
                }
            }
        }

        // 3. Setup audio capture -> WebSocket + NoiseMeter RMS feed
        //    Phase 5: Binary frame optimization — raw PCM without Base64 encoding
        audioCapture.onAudioCaptured = { [weak self] pcmData in
            guard let self else { return }
            let msg = UpstreamMessage.audio(data: pcmData)
            if let binaryFrame = msg.toBinary() {
                self.webSocketManager.sendBinary(binaryFrame)
            } else {
                self.webSocketManager.sendText(msg.toJSON())
            }
        }
        audioCapture.onAudioLevelUpdate = { [weak self] rms in
            guard let self else { return }
            self.sensorManager.processAudioRMS(rms)
            // Normalize RMS (typical range 0~0.3) to 0~1 for aurora visualizer
            let normalized = CGFloat(min(rms / 0.25, 1.0))
            DispatchQueue.main.async {
                self.userAudioEnergy = normalized
            }
        }

        // Wire pipeline error callbacks to developer console
        SileroVAD.shared.onVADError = { [weak self] error in
            DispatchQueue.main.async {
                self?.devConsoleModel.captureNetworkMessage(direction: "ERR", payload: "VAD: \(error.rawValue)")
            }
        }
        audioPlayback.onBufferOverflow = { [weak self] dropped in
            DispatchQueue.main.async {
                self?.devConsoleModel.captureNetworkMessage(direction: "ERR", payload: "Audio overflow: dropped \(dropped) chunks")
            }
        }
        audioPlayback.onPlaybackDrained = { [weak self] in
            self?.webSocketManager.sendText(
                UpstreamMessage.playbackDrained.toJSON()
            )
        }

        // 4. Start sensor collection
        sensorManager.startAll()

        // 5. Start telemetry aggregator
        telemetryAggregator.start(sensorManager: sensorManager, webSocket: webSocketManager)
    }

    /// Start audio-only capture on session ready.
    /// Camera is deferred until the user explicitly activates it (horizontal swipe).
    func startAudioCapture() {
        Task {
            let micGranted = await mediaPermissionGate.ensureMicPermission()
            guard micGranted else {
                await MainActor.run {
                    presentPermissionPrompt(
                        reason: .microphoneRequired,
                        status: "Enable microphone in Settings",
                        transcript: "Please enable microphone permission.",
                        announcement: "Microphone permission required. Use Open Settings button to grant access."
                    )
                }
                logger.error("Audio capture blocked: microphone permission denied")
                return
            }

            await MainActor.run {
                // Start shared engine first — enables hardware AEC between capture & playback.
                SharedAudioEngine.shared.setup()
                audioPlayback.setup()
                thinkingSoundManager.setup()
                if !isMuted {
                    audioCapture.startCapture()
                }
                // Camera is NOT started here — user activates via horizontal swipe.
            }
        }
    }

    /// Toggle camera on/off. Triggered by horizontal swipe gesture.
    func toggleCamera() {
        guard isActive, !isPaused else { return }

        if pendingCameraActivation {
            pendingCameraActivation = false
            connectionStatus = "Connected"
            showToast("Camera activation canceled")
            return
        }

        if isCameraActive {
            // Turn off
            pendingCameraActivation = false
            cameraManager.stopCapture()
            isCameraActive = false
            HapticManager.shared.cameraOff()
            webSocketManager.sendText(UpstreamMessage.cameraToggle(active: false).toJSON())
            UIAccessibility.post(notification: .announcement, argument: "Camera off")
            logger.info("Camera deactivated by user")
        } else {
            // Turn on — check permission first
            Task {
                let camGranted = await mediaPermissionGate.ensureCamPermission()
                guard camGranted else {
                    await MainActor.run {
                        presentPermissionPrompt(
                            reason: .cameraRequired,
                            status: "Camera permission required",
                            transcript: "Camera permission required. Enable in Settings.",
                            announcement: "Camera permission required. Use Open Settings button to grant access."
                        )
                    }
                    logger.error("Camera activation blocked: permission denied")
                    return
                }
                await MainActor.run {
                    switch depthModelState {
                    case .loading:
                        pendingCameraActivation = true
                        let message = cameraPreparingMessage()
                        connectionStatus = message
                        transcript = message
                        showToast(message)
                        UIAccessibility.post(notification: .announcement, argument: message)
                    case .ready:
                        activateCamera(announceDepthUnavailable: false)
                    case .unavailable:
                        activateCamera(announceDepthUnavailable: true)
                    }
                }
            }
        }
    }

    func enterSafeMode() {
        isSafeMode = true
        currentLOD = 1
        connectionStatus = "Safe Mode - Reconnecting..."
        debugModel.isSafeMode = true
        debugModel.currentLOD = 1
        telemetryAggregator.pause()
        HapticManager.shared.safeMode()

        // Local TTS alert (no network dependency)
        let message = connectionLostMessage()
        speakLocalStatus(message)
        UIAccessibility.post(notification: .announcement, argument: message)
        logger.warning("Entered safe mode (LOD 1)")
    }

    func exitSafeMode() {
        isSafeMode = false
        connectionStatus = "Connected"
        debugModel.isSafeMode = false
        telemetryAggregator.resume()

        // Local TTS confirmation
        let message = connectionRestoredMessage()
        speakLocalStatus(message)
        UIAccessibility.post(notification: .announcement, argument: message)
        logger.info("Exited safe mode")
    }

    func clearFaceLibrary() {
        webSocketManager.sendText(UpstreamMessage.clearFaceLibrary.toJSON())
    }

    func switchToUser(_ userId: String) {
        guard userId != SightLineConfig.defaultUserId else { return }
        teardownPipeline()
        connectionStatus = "Switching to \(userId)..."
        logger.info("Switching user to \(userId)")
        // Brief delay to allow teardown to complete before re-setup
        Task {
            try? await Task.sleep(nanoseconds: 200_000_000) // 200ms
            await MainActor.run {
                SightLineConfig.defaultUserId = userId
                SightLineConfig.defaultSessionId = UUID().uuidString.lowercased()
                sessionResumptionHandle = ""
                UserDefaults.standard.removeObject(forKey: SightLineConfig.sessionResumptionHandleDefaultsKey)
                setupPipeline()
            }
        }
    }

    func teardownPipeline() {
        cancelSessionReadyTimeout()
        pendingCameraActivation = false
        telemetryAggregator.stop()
        sensorManager.stopAll()
        audioCapture.stopCapture()
        cameraManager.stopCapture()
        isCameraActive = false
        thinkingSoundManager.teardown()
        audioPlayback.teardown()
        SharedAudioEngine.shared.teardown()
        webSocketManager.disconnect()
    }
}

@MainActor
final class MediaPermissionGate: ObservableObject {
    private enum PermissionStatus {
        case granted
        case denied
        case undetermined
    }

    private let launchArguments = Set(ProcessInfo.processInfo.arguments)

    /// Check/request microphone permission only.
    func ensureMicPermission() async -> Bool {
        return await ensureMicrophonePermission()
    }

    /// Check/request camera permission only.
    func ensureCamPermission() async -> Bool {
        return await ensureCameraPermission()
    }

    func isCameraPermissionDenied() -> Bool {
        currentCameraPermissionStatus() == .denied
    }

    private func ensureCameraPermission() async -> Bool {
        switch currentCameraPermissionStatus() {
        case .granted:
            return true
        case .undetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .denied:
            return false
        }
    }

    private func ensureMicrophonePermission() async -> Bool {
        switch currentMicrophonePermissionStatus() {
        case .granted:
            return true
        case .undetermined:
            return await withCheckedContinuation { continuation in
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .denied:
            return false
        }
    }

    private func currentCameraPermissionStatus() -> PermissionStatus {
        if launchArguments.contains("-uitest-camera-granted") {
            return .granted
        }
        if launchArguments.contains("-uitest-camera-denied") {
            return .denied
        }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return .granted
        case .notDetermined:
            return .undetermined
        case .denied, .restricted:
            return .denied
        @unknown default:
            return .denied
        }
    }

    private func currentMicrophonePermissionStatus() -> PermissionStatus {
        if launchArguments.contains("-uitest-mic-granted") {
            return .granted
        }
        if launchArguments.contains("-uitest-mic-denied") {
            return .denied
        }

        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            return .granted
        case .undetermined:
            return .undetermined
        case .denied:
            return .denied
        @unknown default:
            return .denied
        }
    }
}
