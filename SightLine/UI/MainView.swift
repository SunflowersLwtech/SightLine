//
//  MainView.swift
//  SightLine
//
//  Full-screen dark UI with minimal visual elements.
//  Background color shifts with LOD level. A breathing circle indicates active state.
//  Designed for accessibility - all elements have VoiceOver labels.
//
//  Phase 2: Integrated SensorManager, TelemetryAggregator, and disconnection degradation.
//

import SwiftUI
import AVFoundation
import Combine
import os

private let logger = Logger(subsystem: "com.sightline.app", category: "MainView")

private enum DepthModelState {
    case loading
    case ready
    case unavailable
}

private enum PermissionPromptReason {
    case microphoneRequired
    case cameraRequired
}

struct MainView: View {
    @StateObject private var webSocketManager = WebSocketManager()
    @StateObject private var audioCapture = AudioCaptureManager()
    @StateObject private var audioPlayback = AudioPlaybackManager()
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var frameSelector = FrameSelector()
    @StateObject private var sensorManager = SensorManager()
    @StateObject private var telemetryAggregator = TelemetryAggregator()
    @StateObject private var mediaPermissionGate = MediaPermissionGate()
    @StateObject private var debugModel = DebugOverlayModel()
    @StateObject private var devConsoleModel = DeveloperConsoleModel()

    @State private var transcript: String = ""
    @State private var isActive = false
    @State private var currentLOD: Int = 2
    @State private var connectionStatus: String = "Connecting..."
    @State private var isSafeMode = false
    @State private var whenIdleToolQueue: [String] = []
    @State private var showDevConsole = false
    @State private var showProfileSettings = false
    @State private var isMuted = false
    @State private var isCameraActive = false
    @State private var isPaused = false
    @State private var userAudioEnergy: CGFloat = 0
    @State private var lastAgentTranscript = ""
    @State private var toastMessage: String?
    @State private var isVoiceOverActive = UIAccessibility.isVoiceOverRunning
    @State private var showPermissionPrompt = false
    @State private var permissionPromptReason: PermissionPromptReason?
    @State private var showInteractionGuide = false
    @State private var sessionReadyTimeoutTask: Task<Void, Never>?
    @State private var hasReceivedSessionReady = false
    @State private var depthModelState: DepthModelState = .loading
    @State private var pendingCameraActivation = false
    @State private var sessionResumptionHandle = UserDefaults.standard.string(
        forKey: SightLineConfig.sessionResumptionHandleDefaultsKey
    ) ?? ""

    /// Local TTS synthesizer for disconnection alerts (no network needed).
    private let localSynthesizer = AVSpeechSynthesizer()

    var body: some View {
        ZStack {
            // Background color shifts with LOD level (0.3s smooth transition per spec)
            lodBackgroundColor
                .ignoresSafeArea()
                .animation(.easeInOut(duration: 0.3), value: currentLOD)

            mainContent

            // Floating overlays (toast + DEV button)
            toastOverlay
            devConsoleButton
        }
        // MARK: - Gesture Recognizers
        // Triple tap: repeat last agent sentence
        .onTapGesture(count: 3) {
            handleTripleTap()
        }
        // Double tap: force interrupt agent speech
        .onTapGesture(count: 2) {
            handleDoubleTap()
        }
        // Single tap: toggle mute/unmute microphone
        .onTapGesture(count: 1) {
            handleSingleTap()
        }
        // Long press (3s): emergency pause
        .simultaneousGesture(
            LongPressGesture(minimumDuration: 3.0)
                .onEnded { _ in
                    handleLongPress()
                }
        )
        // Swipe up/down: LOD upgrade/downgrade
        .simultaneousGesture(
            DragGesture(minimumDistance: 50)
                .onEnded { value in
                    handleSwipe(translation: value.translation)
                }
        )
        .onAppear {
            HapticManager.shared.prepare()
            setupPipeline()
            devConsoleModel.bind(
                sensorManager: sensorManager,
                cameraManager: cameraManager,
                debugModel: debugModel,
                frameSelector: frameSelector
            )
        }
        .onDisappear {
            teardownPipeline()
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIAccessibility.voiceOverStatusDidChangeNotification
        )) { _ in
            isVoiceOverActive = UIAccessibility.isVoiceOverRunning
        }
        .onReceive(NotificationCenter.default.publisher(for: .faceLibraryChanged)) { _ in
            webSocketManager.sendText(UpstreamMessage.reloadFaceLibrary.toJSON())
            logger.info("Face library changed notification received, sending reload request")
        }
        // Recheck permissions when returning from Settings
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.didBecomeActiveNotification
        )) { _ in
            if showPermissionPrompt {
                recheckPermissionsAfterSettings()
            }
        }
        .sheet(isPresented: $showDevConsole) {
            DeveloperConsoleView(
                model: devConsoleModel,
                webSocketManager: webSocketManager,
                cameraManager: cameraManager,
                isMuted: $isMuted,
                isPaused: $isPaused
            )
        }
        .sheet(isPresented: $showProfileSettings) {
            ProfileSettingsView(webSocketManager: webSocketManager, onSwitchUser: { userId in
                switchToUser(userId)
            })
        }
        .sheet(isPresented: $showInteractionGuide) {
            InteractionGuideView()
        }
        .accessibilityLabel(buildAccessibilityDescription())
        .accessibilityAction(.default) {
            handleSingleTap()
        }
        .accessibilityHint(isVoiceOverActive
            ? "Double tap to toggle mute. Use Actions rotor for more controls."
            : "")
        .accessibilityAction(named: "Interrupt Speech") {
            handleDoubleTap()
        }
        .accessibilityAction(named: "Repeat Last Message") {
            handleTripleTap()
        }
        .accessibilityAction(named: isPaused ? "Resume Session" : "Pause Session") {
            handleLongPress()
        }
        .accessibilityAction(named: "Increase Detail Level") {
            handleSwipe(translation: CGSize(width: 0, height: -100))
        }
        .accessibilityAction(named: "Decrease Detail Level") {
            handleSwipe(translation: CGSize(width: 0, height: 100))
        }
        .accessibilityAction(named: isCameraActive ? "Turn Off Camera" : "Turn On Camera") {
            toggleCamera()
        }
    }

    // MARK: - Main Content

    private var mainContent: some View {
        VStack(spacing: 0) {
            // Camera preview — integrated in content flow at natural aspect ratio
            if isCameraActive, let session = cameraManager.previewSession {
                cameraSection(session: session)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }

            Spacer()

            // Aurora visualizer (Gemini Live-style audio energy feedback)
            AuroraVisualizer(
                userEnergy: userAudioEnergy,
                modelSpeaking: audioPlayback.isPlaying
            )
            .frame(height: 220)
            .padding(.bottom, -40)

            // Bottom info cluster
            bottomInfoCluster

            // Bottom toolbar: profile button
            HStack {
                Button(action: { showInteractionGuide = true }) {
                    Image(systemName: "questionmark.circle")
                        .font(.system(size: 22))
                        .foregroundColor(.white.opacity(0.5))
                        .padding(12)
                }
                .accessibilityLabel("Open controls guide")
                .accessibilityIdentifier("main-open-guide")

                Spacer()
                Button(action: { showProfileSettings = true }) {
                    Image(systemName: "person.crop.circle")
                        .font(.system(size: 22))
                        .foregroundColor(.white.opacity(0.5))
                        .padding(12)
                }
                .accessibilityLabel("Open settings")
                .accessibilityIdentifier("main-open-settings")
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 16)
        }
        .animation(.easeInOut(duration: 0.3), value: isCameraActive)
    }

    // MARK: - Camera Section

    private func cameraSection(session: AVCaptureSession) -> some View {
        CameraPreviewView(session: session)
            .aspectRatio(3.0 / 4.0, contentMode: .fit)
            .clipShape(.rect(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
            )
            #if DEBUG
            .overlay(
                ZStack {
                    boundingBoxOverlay(boxes: devConsoleModel.visionBoxes, color: .green)
                    boundingBoxOverlay(boxes: devConsoleModel.ocrBoxes, color: .yellow)
                    boundingBoxOverlay(boxes: devConsoleModel.faceBoxes, color: .cyan)
                }
                .clipShape(.rect(cornerRadius: 16))
            )
            #endif
            .overlay(alignment: .topTrailing) {
                cameraFlipButton
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .accessibilityHidden(true)
    }

    private var cameraFlipButton: some View {
        Button {
            cameraManager.flipCamera()
            HapticManager.shared.swipe()
            devConsoleModel.captureTranscript(
                text: "Camera flipped to \(cameraManager.cameraPosition == .back ? "front" : "back")",
                role: "gesture"
            )
        } label: {
            Image(systemName: "arrow.triangle.2.circlepath.camera")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.white.opacity(0.8))
                .padding(8)
                .background(.ultraThinMaterial, in: Circle())
        }
        .padding(10)
        .accessibilityLabel("Flip camera")
    }

    // MARK: - Bottom Info Cluster

    private var bottomInfoCluster: some View {
        VStack(spacing: 8) {
            Text(connectionStatus)
                .font(.caption)
                .foregroundColor(isSafeMode ? .red.opacity(0.7) : .white.opacity(0.5))
                .accessibilityLabel(connectionStatus)
                .accessibilityIdentifier("main-connection-status")

            statusStrip

            if !transcript.isEmpty {
                Text(transcript)
                    .font(.body)
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .lineLimit(3)
                    .accessibilityLabel("Last message: \(transcript)")
            }

            if showPermissionPrompt {
                permissionPromptButton
            }
        }
        .padding(.bottom, 16)
    }

    // MARK: - Toast Overlay

    private var toastOverlay: some View {
        Group {
            if let toast = toastMessage {
                VStack {
                    Spacer()
                    Text(toast)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.ultraThinMaterial)
                        .clipShape(.capsule)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                .padding(.bottom, 120)
                .allowsHitTesting(false)
                .animation(.easeInOut(duration: 0.25), value: toastMessage)
            }
        }
    }

    // MARK: - Dev Console Button

    private var devConsoleButton: some View {
        Group {
            #if DEBUG
            VStack {
                HStack {
                    Spacer()
                    Button(action: { showDevConsole = true }) {
                        HStack(spacing: 4) {
                            Image(systemName: "ant.fill")
                                .font(.system(size: 11))
                            Text("DEV")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                        }
                        .foregroundColor(.green)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial)
                        .clipShape(.capsule)
                        .overlay(
                            Capsule()
                                .stroke(Color.green.opacity(0.3), lineWidth: 0.5)
                        )
                    }
                    .accessibilityLabel("Developer Console")
                }
                .padding(.trailing, 16)
                Spacer()
            }
            .padding(.top, 54)
            #endif
        }
    }

    // MARK: - Gesture Handlers

    private func handleSingleTap() {
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
        logger.info("Gesture: mute_toggle (isMuted=\(isMuted))")
    }

    private func handleDoubleTap() {
        HapticManager.shared.doubleTap()
        audioPlayback.stopImmediately()
        audioPlayback.suppressIncomingAudio(for: 0.8)
        webSocketManager.sendText(UpstreamMessage.gesture(type: "interrupt").toJSON())
        UIAccessibility.post(notification: .announcement, argument: "Speech interrupted")
        devConsoleModel.captureTranscript(text: "Gesture: interrupt", role: "gesture")
        showToast("Speech interrupted")
        logger.info("Gesture: interrupt")
    }

    private func handleTripleTap() {
        HapticManager.shared.tripleTap()
        webSocketManager.sendText(UpstreamMessage.gesture(type: "repeat_last").toJSON())
        UIAccessibility.post(notification: .announcement, argument: "Repeating last message")
        devConsoleModel.captureTranscript(text: "Gesture: repeat_last", role: "gesture")
        showToast("Repeating last message")
        logger.info("Gesture: repeat_last")
    }

    private func handleLongPress() {
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
        logger.info("Gesture: pause (paused=\(isPaused))")
    }

    private func handleSwipe(translation: CGSize) {
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

    // MARK: - LOD Background Colors

    private var lodBackgroundColor: Color {
        switch currentLOD {
        case 1: return Color(red: 0.05, green: 0.05, blue: 0.15)  // Deep blue - silent
        case 2: return Color(red: 0.15, green: 0.10, blue: 0.05)  // Warm orange tint
        case 3: return Color(red: 0.10, green: 0.10, blue: 0.10)  // Soft grey
        default: return .black
        }
    }

    // MARK: - Permission Prompt

    private var permissionPromptButton: some View {
        Button(action: { openAppSettings() }) {
            HStack(spacing: 6) {
                Image(systemName: "gear")
                Text("Open Settings")
            }
            .font(.subheadline.weight(.medium))
            .foregroundColor(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color.blue.opacity(0.7))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .accessibilityLabel("Open Settings to grant permissions")
        .accessibilityHint("Opens iOS Settings where you can enable camera and microphone access for SightLine")
        .accessibilityIdentifier("main-open-app-settings")
        .padding(.top, 4)
    }

    // MARK: - Status Strip

    private var statusStrip: some View {
        HStack(spacing: 6) {
            statusBadge(
                icon: nil,
                text: "LOD \(currentLOD)",
                color: lodBadgeColor,
                accessibilityLabel: "Detail level \(currentLOD)",
                accessibilityHint: isVoiceOverActive ? "Swipe up or down to adjust" : "Swipe up or down to change"
            )
            .accessibilityValue("\(currentLOD) of 3, \(lodName)")
            .accessibilityAdjustableAction { direction in
                switch direction {
                case .increment:
                    handleSwipe(translation: CGSize(width: 0, height: -100))
                case .decrement:
                    handleSwipe(translation: CGSize(width: 0, height: 100))
                @unknown default: break
                }
            }

            if isMuted {
                statusBadge(
                    icon: "mic.slash",
                    text: "Muted",
                    color: .red,
                    accessibilityLabel: "Microphone muted",
                    accessibilityHint: isVoiceOverActive ? "Double tap to unmute" : "Single tap to unmute"
                )
            }

            if isPaused {
                statusBadge(
                    icon: "pause.circle.fill",
                    text: "Paused",
                    color: .orange,
                    accessibilityLabel: "Emergency pause active",
                    accessibilityHint: isVoiceOverActive ? "Use Actions rotor to resume" : "Long press to resume"
                )
            }

            if isCameraActive {
                statusBadge(
                    icon: "camera.fill",
                    text: "Cam",
                    color: .green,
                    accessibilityLabel: "Camera active",
                    accessibilityHint: isVoiceOverActive ? "Use Actions rotor to turn off" : "Swipe sideways to turn off"
                )
            }
        }
        .allowsHitTesting(false)
        .animation(.easeInOut(duration: 0.2), value: isMuted)
        .animation(.easeInOut(duration: 0.2), value: isPaused)
        .animation(.easeInOut(duration: 0.2), value: isCameraActive)
        .animation(.easeInOut(duration: 0.2), value: currentLOD)
    }

    private func statusBadge(
        icon: String?,
        text: String,
        color: Color,
        accessibilityLabel: String,
        accessibilityHint: String
    ) -> some View {
        HStack(spacing: 3) {
            if let icon = icon {
                Image(systemName: icon)
                    .font(.system(size: 9))
            }
            Text(text)
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundColor(.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.25))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(color.opacity(0.5), lineWidth: 1)
        )
        .cornerRadius(6)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint(accessibilityHint)
    }

    private var lodBadgeColor: Color {
        switch currentLOD {
        case 1: return .red
        case 2: return .orange
        case 3: return .green
        default: return .gray
        }
    }

    private var lodName: String {
        switch currentLOD {
        case 1: return "Safety"
        case 2: return "Balanced"
        case 3: return "Detailed"
        default: return "Unknown"
        }
    }

    private func buildAccessibilityDescription() -> String {
        var parts = ["SightLine is \(isActive ? "active" : "connecting")"]
        parts.append("detail level \(currentLOD)")
        if isMuted { parts.append("microphone muted") }
        if isCameraActive { parts.append("camera on") }
        if isPaused { parts.append("session paused") }
        if isVoiceOverActive { parts.append("Use the Actions rotor for controls") }
        return parts.joined(separator: ", ")
    }

    private func currentPreferredLanguage() -> String {
        UserDefaults.standard.string(forKey: SightLineConfig.preferredLanguageDefaultsKey) ?? "en-US"
    }

    private func localizedText(
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

    private func speakLocalStatus(_ message: String) {
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: currentPreferredLanguage())
            ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        localSynthesizer.speak(utterance)
    }

    private func cameraPreparingMessage() -> String {
        localizedText(
            english: "Preparing camera analysis. Please wait.",
            chinese: "正在准备相机分析，请稍候。",
            spanish: "Preparando el analisis de la camara. Espera un momento.",
            japanese: "カメラ解析を準備しています。少しお待ちください。"
        )
    }

    private func cameraDepthUnavailableMessage() -> String {
        localizedText(
            english: "Depth alerts are unavailable. Camera descriptions are still on.",
            chinese: "深度提醒暂时不可用，但相机描述仍然可用。",
            spanish: "Las alertas de profundidad no estan disponibles, pero la camara sigue describiendo la escena.",
            japanese: "深度アラートは利用できませんが、カメラの説明は引き続き使えます。"
        )
    }

    private func sessionReadyTimedOutMessage() -> String {
        localizedText(
            english: "Session initialization timed out. Reconnecting now.",
            chinese: "会话初始化超时，正在重新连接。",
            spanish: "La inicializacion de la sesion agoto el tiempo. Reconectando ahora.",
            japanese: "セッションの初期化がタイムアウトしました。再接続しています。"
        )
    }

    private func connectionLostMessage() -> String {
        localizedText(
            english: "Connection lost. Safe mode active.",
            chinese: "连接已断开，已进入安全模式。",
            spanish: "Se perdio la conexion. El modo seguro esta activo.",
            japanese: "接続が切れました。セーフモードを有効にしました。"
        )
    }

    private func connectionRestoredMessage() -> String {
        localizedText(
            english: "Connection restored.",
            chinese: "连接已恢复。",
            spanish: "Conexion restablecida.",
            japanese: "接続が回復しました。"
        )
    }

    private func capabilityDegradedMessage(for capability: String) -> String {
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

    private func scheduleSessionReadyTimeout() {
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

    private func cancelSessionReadyTimeout() {
        sessionReadyTimeoutTask?.cancel()
        sessionReadyTimeoutTask = nil
    }

    private func activateCamera(announceDepthUnavailable: Bool) {
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

    private func showToast(_ message: String) {
        toastMessage = message
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            if toastMessage == message {
                toastMessage = nil
            }
        }
    }

    private var shouldCaptureDevEvents: Bool {
        #if DEBUG
        return showDevConsole
        #else
        return false
        #endif
    }

    private func openAppSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    private func presentPermissionPrompt(
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

    private func dismissPermissionPrompt() {
        showPermissionPrompt = false
        permissionPromptReason = nil
    }

    /// Recheck permissions when returning from Settings. Dismiss prompt if granted.
    /// If WebSocket was blocked due to missing mic permission, connect now.
    private func recheckPermissionsAfterSettings() {
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

    // MARK: - Bounding Box Overlay

    #if DEBUG
    private func boundingBoxOverlay(
        boxes: [DeveloperConsoleModel.DebugBoundingBox],
        color: Color
    ) -> some View {
        GeometryReader { geometry in
            ForEach(Array(boxes.enumerated()), id: \.element.id) { _, box in
                let rect = box.normalizedRect
                let x = rect.minX * geometry.size.width
                let y = rect.minY * geometry.size.height
                let w = rect.width * geometry.size.width
                let h = rect.height * geometry.size.height
                let labelText = box.confidence > 0
                    ? "\(box.label) \(String(format: "%.2f", box.confidence))"
                    : box.label

                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(color, lineWidth: 2)
                        .frame(width: max(w, 2), height: max(h, 2))

                    Text(labelText)
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(.black)
                        .padding(.horizontal, 3)
                        .padding(.vertical, 1)
                        .background(color.opacity(0.9))
                        .clipShape(.rect(cornerRadius: 2))
                        .offset(x: 0, y: -12)
                }
                .position(x: x + (w / 2), y: y + (h / 2))
            }
        }
        .allowsHitTesting(false)
    }
    #endif

    // MARK: - Pipeline Setup

    private func setupPipeline() {
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

        webSocketManager.onTextSent = { text in
            DispatchQueue.main.async {
                if shouldCaptureDevEvents {
                    devConsoleModel.captureNetworkMessage(direction: "UP", payload: text)
                }
            }
        }

        webSocketManager.onAudioReceived = { [weak audioPlayback, weak audioCapture] data in
            audioCapture?.lastModelAudioReceivedAt = CFAbsoluteTimeGetCurrent()
            audioPlayback?.playAudioData(data)
        }

        audioCapture.isModelAudioPlaying = { [weak audioPlayback] in
            audioPlayback?.isPlaying ?? false
        }

        audioCapture.onVoiceBargeIn = { [weak audioPlayback, weak webSocketManager] in
            DispatchQueue.main.async {
                audioPlayback?.stopImmediately()
                // Short suppression to drop in-flight chunks; the server
                // will stop forwarding once it processes client_barge_in.
                audioPlayback?.suppressIncomingAudio(for: 0.5)
            }
            // With NO_INTERRUPTION, Gemini's server-side VAD won't interrupt
            // the model.  Instead, notify the server so it stops forwarding
            // audio until the current turn completes.
            webSocketManager?.sendText(UpstreamMessage.clientBargeIn.toJSON())
        }

        // Manual activity_start/activity_end signals removed:
        // Server-side VAD is re-enabled with conservative sensitivity.
        // Client-side RMS gating acts as AEC; explicit signals are unnecessary
        // and conflict with automatic activity detection.

        webSocketManager.onTextReceived = { text in
            DispatchQueue.main.async {
                if shouldCaptureDevEvents {
                    devConsoleModel.captureNetworkMessage(direction: "DOWN", payload: text)
                }
            }
            if let msg = DownstreamMessage.parse(text: text) {
                handleDownstreamMessage(msg)
            }
        }

        webSocketManager.onConnectionStateChanged = { connected in
            DispatchQueue.main.async {
                isActive = connected
                debugModel.isConnected = connected
                if connected {
                    connectionStatus = "Connected"
                    scheduleSessionReadyTimeout()
                } else {
                    cancelSessionReadyTimeout()
                    audioCapture.stopCapture()
                    cameraManager.stopCapture()
                    audioPlayback.stopImmediately()
                    isCameraActive = false
                    pendingCameraActivation = false
                }
            }
        }

        // SL-38: Disconnection degradation callbacks
        webSocketManager.onDisconnectionDegraded = {
            DispatchQueue.main.async {
                enterSafeMode()
            }
        }

        webSocketManager.onConnectionRestored = {
            DispatchQueue.main.async {
                exitSafeMode()
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
        cameraManager.onCameraFailure = { reason in
            webSocketManager.sendText(UpstreamMessage.cameraFailure(error: reason).toJSON())
        }
        cameraManager.frameSelector = frameSelector
        cameraManager.onFrameCaptured = { jpegData in
            guard frameSelector.isFrameDifferent(jpegData: jpegData) else {
                frameSelector.markFrameSkipped()
                logger.debug("Frame skipped (pixel-diff below threshold)")
                return
            }
            // Phase 5: Binary frame optimization — eliminates ~33% Base64 overhead
            let msg = UpstreamMessage.image(data: jpegData, mimeType: "image/jpeg")
            if let binaryFrame = msg.toBinary() {
                webSocketManager.sendBinary(binaryFrame)
            } else {
                webSocketManager.sendText(msg.toJSON())
            }
            frameSelector.markFrameSent()
        }

        // 2b. Setup depth estimation pipeline (CoreML Depth Anything V2)
        depthModelState = .loading
        pendingCameraActivation = false
        let depthEstimator = DepthEstimator()
        cameraManager.depthEstimator = depthEstimator
        cameraManager.onDepthEstimated = { summary in
            sensorManager.updateDepth(summary)
        }
        Task(priority: .userInitiated) {
            let available = await Task.detached(priority: .userInitiated) {
                depthEstimator.loadModel()
                return depthEstimator.isAvailable
            }.value

            await MainActor.run {
                depthModelState = available ? .ready : .unavailable
                logger.info("Depth model async load complete (available: \(available))")

                guard pendingCameraActivation else { return }
                if available {
                    connectionStatus = "Connected"
                    activateCamera(announceDepthUnavailable: false)
                } else {
                    activateCamera(announceDepthUnavailable: true)
                }
            }
        }

        // 3. Setup audio capture -> WebSocket + NoiseMeter RMS feed
        //    Phase 5: Binary frame optimization — raw PCM without Base64 encoding
        audioCapture.onAudioCaptured = { pcmData in
            let msg = UpstreamMessage.audio(data: pcmData)
            if let binaryFrame = msg.toBinary() {
                webSocketManager.sendBinary(binaryFrame)
            } else {
                webSocketManager.sendText(msg.toJSON())
            }
        }
        audioCapture.onAudioLevelUpdate = { rms in
            sensorManager.processAudioRMS(rms)
            // Normalize RMS (typical range 0~0.3) to 0~1 for aurora visualizer
            let normalized = CGFloat(min(rms / 0.25, 1.0))
            DispatchQueue.main.async {
                userAudioEnergy = normalized
            }
        }

        // Wire pipeline error callbacks to developer console
        SileroVAD.shared.onVADError = { error in
            DispatchQueue.main.async {
                devConsoleModel.captureNetworkMessage(direction: "ERR", payload: "VAD: \(error.rawValue)")
            }
        }
        audioPlayback.onBufferOverflow = { dropped in
            DispatchQueue.main.async {
                devConsoleModel.captureNetworkMessage(direction: "ERR", payload: "Audio overflow: dropped \(dropped) chunks")
            }
        }
        audioPlayback.onPlaybackDrained = { [weak webSocketManager] in
            webSocketManager?.sendText(
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
    private func startAudioCapture() {
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
                if !isMuted {
                    audioCapture.startCapture()
                }
                // Camera is NOT started here — user activates via horizontal swipe.
            }
        }
    }

    /// Toggle camera on/off. Triggered by horizontal swipe gesture.
    private func toggleCamera() {
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

    private func handleDownstreamMessage(_ msg: DownstreamMessage) {
        switch msg {
        case .sessionReady:
            logger.info("Session ready, starting audio capture (camera deferred)")
            DispatchQueue.main.async {
                hasReceivedSessionReady = true
                cancelSessionReadyTimeout()
                // Pre-set model speaking timestamp so silence gate covers
                // the 200-600ms greeting generation delay, preventing
                // ambient noise from reaching Gemini VAD and interrupting the greeting.
                audioCapture.lastModelAudioReceivedAt = CFAbsoluteTimeGetCurrent()
                startAudioCapture()
            }
        case .faceLibraryReloaded(let count):
            DispatchQueue.main.async {
                let message = "Face library reloaded (\(count) faces)."
                transcript = message
                if shouldCaptureDevEvents {
                    devConsoleModel.captureTranscript(text: message, role: "system")
                }
            }
            logger.info("Face library reloaded: \(count)")
        case .faceLibraryCleared(let deletedCount):
            DispatchQueue.main.async {
                let message = "Face library cleared (\(deletedCount) deleted)."
                transcript = message
                if shouldCaptureDevEvents {
                    devConsoleModel.captureTranscript(text: message, role: "system")
                }
            }
            logger.info("Face library cleared: \(deletedCount)")
        case .error(let message):
            DispatchQueue.main.async {
                cancelSessionReadyTimeout()
                connectionStatus = "Server error — retrying..."
                transcript = "Server error: \(message)"
                if shouldCaptureDevEvents {
                    devConsoleModel.captureTranscript(text: "Server error: \(message)", role: "system")
                }
                audioCapture.stopCapture()
                cameraManager.stopCapture()
            }
            logger.error("Server error message: \(message, privacy: .public)")
            webSocketManager.reconnect(afterMs: 3000)
        case .transcript(let text, let role):
            DispatchQueue.main.async {
                if role == "agent" {
                    // Agent transcripts → visible as subtitles + dev console
                    transcript = text
                    lastAgentTranscript = text
                    if shouldCaptureDevEvents {
                        devConsoleModel.captureTranscript(text: text, role: role)
                    }
                    drainWhenIdleToolQueueIfPossible()
                } else {
                    // User / echo transcripts → dev console only (no main UI)
                    if shouldCaptureDevEvents {
                        devConsoleModel.captureTranscript(text: text, role: role == "user" && audioPlayback.isPlaying ? "echo" : role)
                    }
                }
            }
        case .lodUpdate(let lod):
            DispatchQueue.main.async {
                let lodNames = [1: "Safety", 2: "Balanced", 3: "Detailed"]
                if lod != currentLOD {
                    UIAccessibility.post(notification: .announcement, argument: "Detail level \(lod): \(lodNames[lod] ?? "")")
                }
                currentLOD = lod
                frameSelector.updateLOD(lod)
                telemetryAggregator.updateLOD(lod)
                debugModel.currentLOD = lod
            }
        case .toolEvent(let tool, let behavior, let payload):
            let status = (payload["status"] as? String)?.lowercased() ?? ""
            if !status.isEmpty {
                DispatchQueue.main.async {
                    if shouldCaptureDevEvents {
                        devConsoleModel.captureTranscript(text: "Tool \(tool): \(status)", role: "system")
                        devConsoleModel.captureToolActivity(
                            tool: tool, behavior: behavior.rawValue, status: status
                        )
                    }
                }
            }
            switch status {
            case "queued", "invoked", "completed":
                // Keep these in dev console only; avoid freezing user-facing transcript on tool progress text.
                return
            case "error", "unavailable":
                handleToolMessage(text: "Tool \(tool) is unavailable.", behavior: .WHEN_IDLE)
            default:
                handleToolMessage(
                    text: "Tool update: \(tool)",
                    behavior: behavior
                )
            }
        case .visionResult(let summary, let behavior):
            DispatchQueue.main.async { debugModel.markCapabilityReady("vision") }
            // Haptic feedback for safety-critical vision alerts
            if behavior == .INTERRUPT {
                HapticManager.shared.obstacleProximity(distance: 0.3)
            }
            handleToolMessage(
                text: summary.isEmpty ? "Vision analysis updated." : summary,
                behavior: behavior
            )
        case .ocrResult(let summary, let behavior):
            DispatchQueue.main.async { debugModel.markCapabilityReady("ocr") }
            handleToolMessage(
                text: summary.isEmpty ? "OCR result received." : summary,
                behavior: behavior
            )
        case .visionDebug(let data):
            DispatchQueue.main.async {
                if shouldCaptureDevEvents {
                    devConsoleModel.captureVisionDebug(data)
                }
            }
        case .ocrDebug(let data):
            DispatchQueue.main.async {
                if shouldCaptureDevEvents {
                    devConsoleModel.captureOCRDebug(data)
                }
            }
        case .faceDebug(let data):
            DispatchQueue.main.async {
                if shouldCaptureDevEvents {
                    devConsoleModel.captureFaceDebug(data)
                }
            }
        case .frameAck(let frameId, let queuedAgents):
            DispatchQueue.main.async {
                if shouldCaptureDevEvents {
                    devConsoleModel.captureFrameAck(frameId: frameId, queuedAgents: queuedAgents)
                }
            }
        case .navigationResult(let summary, let behavior):
            // Haptic feedback for urgent navigation (LOD 1 safety mode)
            if currentLOD <= 1 && behavior == .INTERRUPT {
                HapticManager.shared.directionalCue(.ahead)
            }
            handleToolMessage(
                text: summary.isEmpty ? "Navigation result received." : summary,
                behavior: behavior,
                isUrgent: currentLOD <= 1
            )
        case .searchResult(let summary, let behavior):
            handleToolMessage(
                text: summary.isEmpty ? "Search result received." : summary,
                behavior: behavior
            )
        case .personIdentified(let name, let behavior):
            DispatchQueue.main.async { debugModel.markCapabilityReady("face") }
            handleIdentityMessage(name: name, matched: true, behavior: behavior)
        case .identityUpdate(let name, let matched, let behavior):
            DispatchQueue.main.async { debugModel.markCapabilityReady("face") }
            handleIdentityMessage(name: name, matched: matched, behavior: behavior)
        case .capabilityDegraded(let capability, let reason, _):
            DispatchQueue.main.async {
                debugModel.markCapabilityDegraded(capability)
                handleToolMessage(
                    text: capabilityDegradedMessage(for: capability),
                    behavior: .WHEN_IDLE
                )
            }
            logger.warning("Capability degraded: \(capability, privacy: .public) - \(reason, privacy: .public)")
        case .debugLod(let data):
            let memoryTop3 = (data["memory_top3"] as? [String]) ?? []
            DispatchQueue.main.async {
                // SL-77: explicit debugLod -> memory top3 injection for overlay gate.
                debugModel.memoryTop3 = Array(memoryTop3.prefix(3))
                debugModel.updateFromLodDebug(data)
            }
        case .debugActivity(let data):
            DispatchQueue.main.async {
                debugModel.updateFromActivityDebug(data)
            }
        case .interrupted:
            logger.info("Model output interrupted — flushing playback buffer")
            DispatchQueue.main.async {
                audioCapture.lastModelAudioReceivedAt = 0  // Expire echo gate immediately
                audioPlayback.stopImmediately()
                audioPlayback.suppressIncomingAudio(for: 0.8)
                HapticManager.shared.doubleTap()
                if shouldCaptureDevEvents {
                    devConsoleModel.captureTranscript(
                        text: "Model interrupted by user", role: "system")
                }
            }
        case .goAway(let retryMs):
            logger.info("GoAway received, reconnecting in \(retryMs)ms")
            DispatchQueue.main.async {
                cancelSessionReadyTimeout()
                connectionStatus = "Reconnecting..."
            }
            webSocketManager.reconnect(afterMs: retryMs)
        case .sessionResumption(let handle):
            guard !handle.isEmpty else { return }
            DispatchQueue.main.async {
                sessionResumptionHandle = handle
            }
            UserDefaults.standard.set(handle, forKey: SightLineConfig.sessionResumptionHandleDefaultsKey)
            logger.info("Session resumption handle updated: \(handle.prefix(20))...")
        case .toolsManifest(let tools, let modules, let agents):
            DispatchQueue.main.async {
                devConsoleModel.captureToolsManifest(tools: tools, modules: modules, subAgents: agents)
            }
        case .profileUpdatedAck:
            logger.info("Profile update acknowledged by server")
        case .unknown(let raw):
            logger.debug("Unknown downstream message: \(String(raw.prefix(200)), privacy: .public)")
            DispatchQueue.main.async {
                if shouldCaptureDevEvents {
                    devConsoleModel.captureTranscript(
                        text: "Unknown downstream: \(String(raw.prefix(160)))",
                        role: "system"
                    )
                }
            }
        default:
            break
        }
    }

    // MARK: - Safe Mode (SL-38)

    private func enterSafeMode() {
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

    private func exitSafeMode() {
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

    // MARK: - Tool Behavior Routing (SL-55)

    private func handleToolMessage(
        text: String,
        behavior: ToolBehaviorMode,
        isUrgent: Bool = false
    ) {
        DispatchQueue.main.async {
            let effectiveBehavior: ToolBehaviorMode
            if behavior == .INTERRUPT && audioPlayback.isPlaying && !isUrgent {
                // Guardrail: non-urgent tool events should not cut an utterance in half.
                effectiveBehavior = .WHEN_IDLE
                logger.info("Downgraded non-urgent INTERRUPT to WHEN_IDLE while audio is playing")
            } else {
                effectiveBehavior = behavior
            }

            switch effectiveBehavior {
            case .INTERRUPT:
                // Urgent INTERRUPT must stop ongoing playback immediately.
                audioPlayback.stopImmediately()
                transcript = text
            case .WHEN_IDLE:
                // WHEN_IDLE respects current playback state and queues updates.
                if audioPlayback.isPlaying {
                    whenIdleToolQueue.append(text)
                } else {
                    transcript = text
                }
            case .SILENT:
                logger.debug("SILENT tool update received")
            }
            drainWhenIdleToolQueueIfPossible()
        }
    }

    private func handleIdentityMessage(name: String, matched: Bool, behavior: ToolBehaviorMode) {
        let personText = matched ? "Person identified: \(name)" : "Identity update available."
        if behavior == .SILENT {
            // identify_person must support SILENT path and avoid hard interruption.
            logger.debug("identity SILENT update for \(name, privacy: .public)")
            return
        }
        handleToolMessage(text: personText, behavior: behavior)
    }

    private func drainWhenIdleToolQueueIfPossible() {
        guard !audioPlayback.isPlaying else { return }
        guard !whenIdleToolQueue.isEmpty else { return }
        transcript = whenIdleToolQueue.last ?? transcript
        whenIdleToolQueue.removeAll()
    }

    // MARK: - Face Privacy Action (SL-59)

    private func clearFaceLibrary() {
        webSocketManager.sendText(UpstreamMessage.clearFaceLibrary.toJSON())
    }

    // MARK: - User Switching

    private func switchToUser(_ userId: String) {
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

    // MARK: - Teardown

    private func teardownPipeline() {
        cancelSessionReadyTimeout()
        pendingCameraActivation = false
        telemetryAggregator.stop()
        sensorManager.stopAll()
        audioCapture.stopCapture()
        cameraManager.stopCapture()
        isCameraActive = false
        audioPlayback.teardown()
        SharedAudioEngine.shared.teardown()
        webSocketManager.disconnect()
    }
}

@MainActor
private final class MediaPermissionGate: ObservableObject {
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
