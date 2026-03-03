//
//  HapticManager.swift
//  SightLine
//
//  Provides distinct haptic patterns for each gesture type, plus
//  advanced Core Haptics patterns for obstacle proximity, directional
//  cues, and object type textures.
//  Designed for visually impaired users who rely on tactile feedback
//  to confirm gesture recognition and sense spatial information.
//

import UIKit
import CoreHaptics

final class HapticManager {
    static let shared = HapticManager()

    // MARK: - Existing UIKit generators (preserved)

    private let lightImpact = UIImpactFeedbackGenerator(style: .light)
    private let mediumImpact = UIImpactFeedbackGenerator(style: .medium)
    private let heavyImpact = UIImpactFeedbackGenerator(style: .heavy)
    private let notification = UINotificationFeedbackGenerator()
    private let selection = UISelectionFeedbackGenerator()

    // MARK: - Core Haptics engine

    private var hapticEngine: CHHapticEngine?
    private var supportsHaptics: Bool = false

    private init() {
        setupCoreHaptics()
    }

    // MARK: - Setup

    /// Pre-warm all generators so first haptic fires without delay.
    func prepare() {
        lightImpact.prepare()
        mediumImpact.prepare()
        heavyImpact.prepare()
        notification.prepare()
        selection.prepare()
    }

    private func setupCoreHaptics() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        supportsHaptics = true

        do {
            hapticEngine = try CHHapticEngine()
            hapticEngine?.stoppedHandler = { [weak self] reason in
                // Auto-restart on stop
                try? self?.hapticEngine?.start()
            }
            hapticEngine?.resetHandler = { [weak self] in
                try? self?.hapticEngine?.start()
            }
            try hapticEngine?.start()
        } catch {
            supportsHaptics = false
        }
    }

    private func ensureEngineRunning() {
        guard supportsHaptics else { return }
        try? hapticEngine?.start()
    }

    // MARK: - Existing gesture feedback (all preserved)

    /// Single tap: light impact confirming mute toggle.
    func singleTap() {
        lightImpact.impactOccurred()
    }

    /// Double tap: two medium impacts confirming speech interrupt.
    func doubleTap() {
        mediumImpact.impactOccurred()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.mediumImpact.impactOccurred()
        }
    }

    /// Triple tap: success notification confirming repeat-last request.
    func tripleTap() {
        notification.notificationOccurred(.success)
    }

    /// Long press: heavy impact confirming emergency pause.
    func longPress() {
        heavyImpact.impactOccurred()
    }

    /// Swipe: selection tick confirming LOD change.
    func swipe() {
        selection.selectionChanged()
    }

    /// Safe mode: three heavy impacts signaling disconnection.
    func safeMode() {
        heavyImpact.impactOccurred()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) { [weak self] in
            self?.heavyImpact.impactOccurred()
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.24) { [weak self] in
            self?.heavyImpact.impactOccurred()
        }
    }

    /// Connection restored: success notification confirming reconnection.
    func connectionRestored() {
        notification.notificationOccurred(.success)
    }

    /// Connection lost: warning notification for disconnection.
    func connectionLost() {
        notification.notificationOccurred(.warning)
    }

    /// Camera on: medium impact simulating a shutter click.
    func cameraOn() {
        mediumImpact.impactOccurred()
    }

    /// Camera off: light impact confirming deactivation.
    func cameraOff() {
        lightImpact.impactOccurred()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) { [weak self] in
            self?.lightImpact.impactOccurred()
        }
    }

    // MARK: - Core Haptics: Obstacle Proximity

    /// Obstacle proximity feedback: intensity scales inversely with distance.
    /// - Parameter distance: Normalized 0.0 (contact) to 1.0 (far away).
    func obstacleProximity(distance: Float) {
        guard supportsHaptics else {
            heavyImpact.impactOccurred()
            return
        }
        ensureEngineRunning()

        let clamped = max(0.0, min(1.0, distance))
        let intensity = 1.0 - clamped
        let sharpness = 1.0 - (clamped * 0.7)

        do {
            let event = CHHapticEvent(
                eventType: .hapticTransient,
                parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness),
                ],
                relativeTime: 0
            )
            let pattern = try CHHapticPattern(events: [event], parameters: [])
            let player = try hapticEngine?.makePlayer(with: pattern)
            try player?.start(atTime: CHHapticTimeImmediate)
        } catch {
            heavyImpact.impactOccurred()
        }
    }

    // MARK: - Core Haptics: Directional Cues

    enum Direction {
        case left, right, ahead, stop
    }

    /// Directional haptic cue for navigation guidance.
    func directionalCue(_ direction: Direction) {
        guard supportsHaptics else {
            mediumImpact.impactOccurred()
            return
        }
        ensureEngineRunning()

        switch direction {
        case .left:
            playPattern(named: "direction_left")
        case .right:
            playPattern(named: "direction_right")
        case .ahead:
            playAheadPattern()
        case .stop:
            playStopPattern()
        }
    }

    private func playAheadPattern() {
        do {
            // Steady pulse: even rhythm
            let events = (0..<3).map { i in
                CHHapticEvent(
                    eventType: .hapticTransient,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.6),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.4),
                    ],
                    relativeTime: Double(i) * 0.15
                )
            }
            let pattern = try CHHapticPattern(events: events, parameters: [])
            let player = try hapticEngine?.makePlayer(with: pattern)
            try player?.start(atTime: CHHapticTimeImmediate)
        } catch {
            mediumImpact.impactOccurred()
        }
    }

    private func playStopPattern() {
        do {
            // Urgent rapid burst: 5 sharp hits
            let events = (0..<5).map { i in
                CHHapticEvent(
                    eventType: .hapticTransient,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: 1.0),
                    ],
                    relativeTime: Double(i) * 0.08
                )
            }
            let pattern = try CHHapticPattern(events: events, parameters: [])
            let player = try hapticEngine?.makePlayer(with: pattern)
            try player?.start(atTime: CHHapticTimeImmediate)
        } catch {
            heavyImpact.impactOccurred()
        }
    }

    // MARK: - Core Haptics: Object Type Texture

    enum ObjectType {
        case person, vehicle, stairs, door, obstacle
    }

    /// Distinctive haptic texture for different object types.
    func objectTexture(_ type: ObjectType) {
        guard supportsHaptics else {
            mediumImpact.impactOccurred()
            return
        }
        ensureEngineRunning()

        do {
            let events: [CHHapticEvent]
            switch type {
            case .person:
                // Gentle pulse
                events = [
                    CHHapticEvent(
                        eventType: .hapticContinuous,
                        parameters: [
                            CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.4),
                            CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.2),
                        ],
                        relativeTime: 0, duration: 0.3
                    ),
                ]
            case .vehicle:
                // Sharp buzz
                events = [
                    CHHapticEvent(
                        eventType: .hapticTransient,
                        parameters: [
                            CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0),
                            CHHapticEventParameter(parameterID: .hapticSharpness, value: 1.0),
                        ],
                        relativeTime: 0
                    ),
                    CHHapticEvent(
                        eventType: .hapticContinuous,
                        parameters: [
                            CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.8),
                            CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.9),
                        ],
                        relativeTime: 0.05, duration: 0.2
                    ),
                ]
            case .stairs:
                // Rhythmic stepping (use AHAP file)
                playPattern(named: "stairs_ahead")
                return
            case .door:
                // Single knock
                events = [
                    CHHapticEvent(
                        eventType: .hapticTransient,
                        parameters: [
                            CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.7),
                            CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.5),
                        ],
                        relativeTime: 0
                    ),
                ]
            case .obstacle:
                // Use proximity AHAP
                playPattern(named: "obstacle_proximity")
                return
            }

            let pattern = try CHHapticPattern(events: events, parameters: [])
            let player = try hapticEngine?.makePlayer(with: pattern)
            try player?.start(atTime: CHHapticTimeImmediate)
        } catch {
            mediumImpact.impactOccurred()
        }
    }

    // MARK: - AHAP Pattern Player

    /// Play a named AHAP haptic pattern from the Haptics resource bundle.
    func playPattern(named name: String) {
        guard supportsHaptics, let engine = hapticEngine else {
            mediumImpact.impactOccurred()
            return
        }
        ensureEngineRunning()

        guard let url = Bundle.main.url(forResource: name, withExtension: "ahap",
                                         subdirectory: "Haptics") else {
            // Fallback: try without subdirectory (flat bundle)
            if let flatURL = Bundle.main.url(forResource: name, withExtension: "ahap") {
                do {
                    try engine.playPattern(from: flatURL)
                } catch {
                    mediumImpact.impactOccurred()
                }
            } else {
                mediumImpact.impactOccurred()
            }
            return
        }

        do {
            try engine.playPattern(from: url)
        } catch {
            mediumImpact.impactOccurred()
        }
    }
}
