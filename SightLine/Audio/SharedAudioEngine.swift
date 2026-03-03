//
//  SharedAudioEngine.swift
//  SightLine
//
//  Single AVAudioEngine shared between capture and playback so that
//  Apple's hardware echo cancellation (AEC) works correctly.
//  Two independent engines cannot share an AEC graph — merging them
//  into one eliminates the model-interrupts-itself echo loop.
//

import AVFoundation
import os

extension Notification.Name {
    static let sharedAudioEngineWillRestart = Notification.Name("sharedAudioEngineWillRestart")
    static let sharedAudioEngineDidRestart = Notification.Name("sharedAudioEngineDidRestart")
    static let sharedAudioEngineDidPause = Notification.Name("sharedAudioEngineDidPause")
    static let audioBufferOverflow = Notification.Name("audioBufferOverflow")
}

final class SharedAudioEngine {
    static let shared = SharedAudioEngine()

    private static let logger = Logger(subsystem: "com.sightline.app", category: "SharedAudioEngine")

    private(set) var engine: AVAudioEngine?
    private(set) var playerNode: AVAudioPlayerNode?
    private(set) var isVoiceProcessingEnabled = false
    private(set) var isRunning = false

    private var routeChangeObserver: NSObjectProtocol?
    private var interruptionObserver: NSObjectProtocol?

    private init() {}

    // MARK: - Lifecycle

    func setup() {
        teardown()

        do {
            try AudioSessionManager.shared.configure()
        } catch {
            Self.logger.error("AudioSession configuration failed: \(error)")
        }

        let newEngine = AVAudioEngine()
        let newPlayer = AVAudioPlayerNode()

        newEngine.attach(newPlayer)

        // Connect player → mainMixer with Gemini's output format (24kHz PCM Int16)
        guard let playbackFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: SightLineConfig.audioOutputSampleRate,
            channels: 1,
            interleaved: true
        ) else {
            Self.logger.error("Failed to create playback format")
            return
        }
        newEngine.connect(newPlayer, to: newEngine.mainMixerNode, format: playbackFormat)

        // Enable hardware voice processing (AEC) on the input node.
        // This is the core fix: with capture and playback on the same engine,
        // the system can cancel speaker audio from the mic signal.
        do {
            try newEngine.inputNode.setVoiceProcessingEnabled(true)
            isVoiceProcessingEnabled = true
            Self.logger.info("Voice processing enabled (hardware AEC active)")
        } catch {
            // Simulator or unsupported hardware — AEC won't work but audio still flows.
            // Server-side Jaccard echo detection remains as fallback.
            isVoiceProcessingEnabled = false
            Self.logger.warning("Voice processing unavailable: \(error). Server echo detection is fallback.")
        }

        newEngine.prepare()
        do {
            try newEngine.start()
            newPlayer.play()

            engine = newEngine
            playerNode = newPlayer
            isRunning = true

            registerNotifications()
            Self.logger.info("SharedAudioEngine started (VP=\(self.isVoiceProcessingEnabled))")
        } catch {
            Self.logger.error("SharedAudioEngine start failed: \(error)")
        }
    }

    func teardown() {
        removeNotifications()

        playerNode?.stop()
        engine?.stop()
        engine = nil
        playerNode = nil
        isRunning = false
        isVoiceProcessingEnabled = false

        Self.logger.info("SharedAudioEngine stopped")
    }

    /// Lightweight stop that preserves notification observers (unlike teardown).
    /// Used during audio interruptions so we can resume when the interruption ends.
    private func pause() {
        playerNode?.stop()
        engine?.stop()
        isRunning = false
        Self.logger.info("SharedAudioEngine paused (interruption began)")
        NotificationCenter.default.post(name: .sharedAudioEngineDidPause, object: nil)
    }

    // MARK: - Route Change Handling

    private func registerNotifications() {
        routeChangeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            self?.handleRouteChange(notification)
        }

        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            self?.handleInterruption(notification)
        }
    }

    private func removeNotifications() {
        if let obs = routeChangeObserver {
            NotificationCenter.default.removeObserver(obs)
            routeChangeObserver = nil
        }
        if let obs = interruptionObserver {
            NotificationCenter.default.removeObserver(obs)
            interruptionObserver = nil
        }
    }

    private func handleRouteChange(_ notification: Notification) {
        guard let info = notification.userInfo,
              let reasonValue = info[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else { return }

        switch reason {
        case .newDeviceAvailable, .oldDeviceUnavailable:
            // Bluetooth connected/disconnected — restart so AEC adapts to new route
            Self.logger.info("Audio route changed (\(reason.rawValue)), restarting engine for AEC re-init")
            restart()
        default:
            break
        }
    }

    private func handleInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else { return }

        switch type {
        case .began:
            Self.logger.info("Audio interruption began (phone call / Siri), pausing engine")
            pause()
        case .ended:
            // Accessibility app — audio MUST resume regardless of shouldResume flag.
            Self.logger.info("Audio interruption ended, restarting engine")
            restart()
        @unknown default:
            Self.logger.warning("Unknown interruption type: \(typeValue)")
        }
    }

    private func restart() {
        NotificationCenter.default.post(name: .sharedAudioEngineWillRestart, object: nil)
        setup()
        NotificationCenter.default.post(name: .sharedAudioEngineDidRestart, object: nil)
    }
}
