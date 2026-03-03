//
//  WatchReceiver.swift
//  SightLine
//
//  Receives real-time heart rate data from the watchOS Companion App
//  via WCSession. This is the PRIMARY heart rate channel (<1s latency).
//  HealthKitManager serves as backup (10-20 min system sync delay).
//
//  Data flow:
//    Apple Watch (WorkoutManager) → WCSession.sendMessage
//    → WatchReceiver.didReceiveMessage → SensorManager.heartRate
//    → TelemetryAggregator → WebSocket → Cloud Run → LOD Engine
//

@preconcurrency import WatchConnectivity
import Foundation
import Combine
import os

class WatchReceiver: NSObject, ObservableObject {
    /// Real-time heart rate from Apple Watch via WCSession.
    /// nil when no watch data has been received.
    @Published var heartRate: Double? = nil

    /// Whether the Apple Watch is reachable via WCSession.
    @Published var isWatchReachable: Bool = false

    /// Whether the watch is actively monitoring (workout session running).
    @Published var isWatchMonitoring: Bool = false

    /// Timestamp of the last received heart rate sample.
    @Published var lastUpdateTime: Date? = nil

    // Watch extended context
    @Published var watchPitch: Double?
    @Published var watchRoll: Double?
    @Published var watchYaw: Double?
    @Published var watchStabilityScore: Double?
    @Published var watchHeading: Double?
    @Published var watchHeadingAccuracy: Double?
    @Published var spO2: Double?
    @Published var watchNoiseExposure: Double?

    private static let logger = Logger(
        subsystem: "com.sightline.app",
        category: "WatchReceiver"
    )

    /// Timeout after which watch heart rate is considered stale (30 seconds).
    private let staleThreshold: TimeInterval = 30.0

    // MARK: - Activation

    /// Activate WCSession on the iPhone side. Must be called once at app launch.
    func activate() {
        guard WCSession.isSupported() else {
            Self.logger.info("WCSession not supported (no paired Apple Watch)")
            return
        }

        WCSession.default.delegate = self
        WCSession.default.activate()
        Self.logger.info("WCSession activation requested (iPhone side)")
    }

    // MARK: - Heart Rate Access

    /// Returns the real-time watch heart rate if fresh (within staleThreshold),
    /// otherwise nil (caller should fall back to HealthKit).
    var freshHeartRate: Double? {
        guard let hr = heartRate, hr > 0,
              let lastUpdate = lastUpdateTime,
              Date().timeIntervalSince(lastUpdate) < staleThreshold else {
            return nil
        }
        return hr
    }

    // MARK: - Private

    private func processWatchMessage(_ payload: [String: Any]) {
        let bpm = payload["heartRate"] as? Double
        let isMonitoring = payload["isMonitoring"] as? Bool ?? (bpm != nil)
        let timestamp = payload["ts"] as? TimeInterval

        // New: motion data
        let motion = payload["motion"] as? [String: Any]
        let pitch = motion?["pitch"] as? Double
        let roll = motion?["roll"] as? Double
        let yaw = motion?["yaw"] as? Double
        let stabilityScore = motion?["stabilityScore"] as? Double

        // New: heading
        let heading = payload["heading"] as? Double
        let headingAccuracy = payload["headingAccuracy"] as? Double

        // New: health data
        let spO2Value = payload["spO2"] as? Double
        let noiseExposure = payload["noiseExposure"] as? Double

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            if isMonitoring, let bpm = bpm, bpm > 0 {
                self.heartRate = bpm
                self.lastUpdateTime = timestamp.map { Date(timeIntervalSince1970: $0) } ?? Date()
                self.isWatchMonitoring = true
            } else if !isMonitoring {
                self.isWatchMonitoring = false
                self.heartRate = nil
                self.lastUpdateTime = nil
            }

            // Motion data
            self.watchPitch = pitch
            self.watchRoll = roll
            self.watchYaw = yaw
            self.watchStabilityScore = stabilityScore

            // Heading
            self.watchHeading = heading
            self.watchHeadingAccuracy = headingAccuracy

            // Health data
            if let spO2Value { self.spO2 = spO2Value }
            if let noiseExposure { self.watchNoiseExposure = noiseExposure }
        }

        if let bpm = bpm, bpm > 0 {
            Self.logger.debug("Watch data received: HR=\(Int(bpm)) BPM, motion=\(motion != nil), heading=\(heading != nil)")
        } else {
            Self.logger.info("Watch payload received (no HR)")
        }
    }
}

// MARK: - WCSessionDelegate

extension WatchReceiver: WCSessionDelegate {

    // Required: activation completion
    func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        if let error = error {
            Self.logger.error("WCSession activation failed: \(error.localizedDescription)")
            return
        }

        Self.logger.info(
            "WCSession activated: state=\(activationState.rawValue), paired=\(session.isPaired), installed=\(session.isWatchAppInstalled)"
        )

        let reachable = session.isReachable
        DispatchQueue.main.async { [weak self] in
            self?.isWatchReachable = reachable
        }
    }

    // Required on iOS: session became inactive (watch switching)
    func sessionDidBecomeInactive(_ session: WCSession) {
        Self.logger.info("WCSession became inactive")
    }

    // Required on iOS: session deactivated (re-activate for new watch)
    func sessionDidDeactivate(_ session: WCSession) {
        Self.logger.info("WCSession deactivated — re-activating")
        WCSession.default.activate()
    }

    // Real-time message from watch (sendMessage path, <1s)
    func session(
        _ session: WCSession,
        didReceiveMessage message: [String: Any]
    ) {
        processWatchMessage(message)
    }

    // Queued transfer from watch (transferUserInfo path, delayed)
    func session(
        _ session: WCSession,
        didReceiveUserInfo userInfo: [String: Any] = [:]
    ) {
        processWatchMessage(userInfo)
    }

    // Reachability changed
    func sessionReachabilityDidChange(_ session: WCSession) {
        let reachable = session.isReachable
        Self.logger.info("Watch reachability changed: \(reachable)")

        DispatchQueue.main.async { [weak self] in
            self?.isWatchReachable = reachable
        }
    }
}
