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

struct MainView: View {
    @StateObject private var coordinator = PipelineCoordinator()
    @State private var showProfileSettings = false
    @State private var isVoiceOverActive = UIAccessibility.isVoiceOverRunning
    @State private var showInteractionGuide = false

    private var webSocketManager: WebSocketManager { coordinator.webSocketManager }
    private var audioCapture: AudioCaptureManager { coordinator.audioCapture }
    private var audioPlayback: AudioPlaybackManager { coordinator.audioPlayback }
    private var cameraManager: CameraManager { coordinator.cameraManager }
    private var frameSelector: FrameSelector { coordinator.frameSelector }
    private var sensorManager: SensorManager { coordinator.sensorManager }
    private var telemetryAggregator: TelemetryAggregator { coordinator.telemetryAggregator }
    private var debugModel: DebugOverlayModel { coordinator.debugModel }
    private var devConsoleModel: DeveloperConsoleModel { coordinator.devConsoleModel }
    private var transcript: String { coordinator.transcript }
    private var isActive: Bool { coordinator.isActive }
    private var currentLOD: Int { coordinator.currentLOD }
    private var connectionStatus: String { coordinator.connectionStatus }
    private var isSafeMode: Bool { coordinator.isSafeMode }
    private var showDevConsole: Bool {
        get { coordinator.showDevConsole }
        nonmutating set { coordinator.showDevConsole = newValue }
    }
    private var isMuted: Bool {
        get { coordinator.isMuted }
        nonmutating set { coordinator.isMuted = newValue }
    }
    private var isCameraActive: Bool { coordinator.isCameraActive }
    private var isPaused: Bool {
        get { coordinator.isPaused }
        nonmutating set { coordinator.isPaused = newValue }
    }
    private var userAudioEnergy: CGFloat { coordinator.userAudioEnergy }
    private var toastMessage: String? { coordinator.toastMessage }
    private var showPermissionPrompt: Bool { coordinator.showPermissionPrompt }
    private var showDevConsoleBinding: Binding<Bool> {
        Binding(
            get: { coordinator.showDevConsole },
            set: { coordinator.showDevConsole = $0 }
        )
    }
    private var isMutedBinding: Binding<Bool> {
        Binding(
            get: { coordinator.isMuted },
            set: { coordinator.isMuted = $0 }
        )
    }
    private var isPausedBinding: Binding<Bool> {
        Binding(
            get: { coordinator.isPaused },
            set: { coordinator.isPaused = $0 }
        )
    }

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
        .sheet(isPresented: showDevConsoleBinding) {
            DeveloperConsoleView(
                model: devConsoleModel,
                webSocketManager: webSocketManager,
                cameraManager: cameraManager,
                isMuted: isMutedBinding,
                isPaused: isPausedBinding
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
        coordinator.handleSingleTap()
    }

    private func handleDoubleTap() {
        coordinator.handleDoubleTap()
    }

    private func handleTripleTap() {
        coordinator.handleTripleTap()
    }

    private func handleLongPress() {
        coordinator.handleLongPress()
    }

    private func handleSwipe(translation: CGSize) {
        coordinator.handleSwipe(translation: translation)
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

    private func openAppSettings() {
        coordinator.openAppSettings()
    }

    /// Recheck permissions when returning from Settings. Dismiss prompt if granted.
    /// If WebSocket was blocked due to missing mic permission, connect now.
    private func recheckPermissionsAfterSettings() {
        coordinator.recheckPermissionsAfterSettings()
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
        coordinator.setupPipeline()
    }

    /// Start audio-only capture on session ready.
    /// Camera is deferred until the user explicitly activates it (horizontal swipe).
    private func startAudioCapture() {
        coordinator.startAudioCapture()
    }

    /// Toggle camera on/off. Triggered by horizontal swipe gesture.
    private func toggleCamera() {
        coordinator.toggleCamera()
    }

    // MARK: - Safe Mode (SL-38)

    private func enterSafeMode() {
        coordinator.enterSafeMode()
    }

    private func exitSafeMode() {
        coordinator.exitSafeMode()
    }

    // MARK: - Face Privacy Action (SL-59)

    private func clearFaceLibrary() {
        coordinator.clearFaceLibrary()
    }

    // MARK: - User Switching

    private func switchToUser(_ userId: String) {
        coordinator.switchToUser(userId)
    }

    // MARK: - Teardown

    private func teardownPipeline() {
        coordinator.teardownPipeline()
    }
}
