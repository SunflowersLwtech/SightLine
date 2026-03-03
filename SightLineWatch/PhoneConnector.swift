//
//  PhoneConnector.swift
//  SightLineWatch
//
//  Manages WCSession from watchOS → iPhone for real-time context delivery.
//  Uses sendMessage when iPhone is reachable (<1s), falls back to
//  transferUserInfo when unreachable (queued for later delivery).
//

@preconcurrency import WatchConnectivity
import Foundation
import Combine
import os

class PhoneConnector: NSObject, ObservableObject {
    static let shared = PhoneConnector()

    @Published var isReachable: Bool = false

    nonisolated private static let logger = Logger(
        subsystem: "com.sightline.watch",
        category: "PhoneConnector"
    )

    private override init() {
        super.init()
    }

    // MARK: - Activation

    /// Activate WCSession. Must be called once at app launch.
    func activate() {
        guard WCSession.isSupported() else {
            Self.logger.warning("WCSession not supported on this device")
            return
        }

        WCSession.default.delegate = self
        WCSession.default.activate()
        Self.logger.info("WCSession activation requested")
    }

    // MARK: - Send Watch Context

    /// Send full watch context to iPhone.
    func sendWatchContext(
        heartRate: Double,
        isMonitoring: Bool = true,
        motion: (pitch: Double, roll: Double, yaw: Double, stabilityScore: Double)? = nil,
        heading: Double? = nil,
        headingAccuracy: Double? = nil,
        spO2: Double? = nil,
        noiseExposure: Double? = nil
    ) {
        var payload: [String: Any] = [
            "heartRate": heartRate,
            "ts": Date().timeIntervalSince1970,
            "isMonitoring": isMonitoring,
        ]

        if let m = motion {
            payload["motion"] = [
                "pitch": m.pitch,
                "roll": m.roll,
                "yaw": m.yaw,
                "stabilityScore": m.stabilityScore,
            ]
        }

        if let h = heading {
            payload["heading"] = h
            if let acc = headingAccuracy {
                payload["headingAccuracy"] = acc
            }
        }

        if let spo2 = spO2 {
            payload["spO2"] = spo2
        }

        if let noise = noiseExposure {
            payload["noiseExposure"] = noise
        }

        let session = WCSession.default
        guard session.activationState == .activated else {
            Self.logger.warning("WCSession not activated, dropping watch context")
            return
        }

        if session.isReachable {
            session.sendMessage(payload, replyHandler: nil) { error in
                Self.logger.error("sendMessage failed: \(error.localizedDescription)")
                session.transferUserInfo(payload)
            }
        } else {
            session.transferUserInfo(payload)
        }
    }

    /// Convenience wrapper for backward compatibility.
    func sendHeartRate(_ bpm: Double, isMonitoring: Bool = true) {
        sendWatchContext(heartRate: bpm, isMonitoring: isMonitoring)
    }
}

// MARK: - WCSessionDelegate

extension PhoneConnector: WCSessionDelegate {
    nonisolated func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        if let error = error {
            Self.logger.error("WCSession activation failed: \(error.localizedDescription)")
            return
        }

        let reachable = session.isReachable
        Self.logger.info("WCSession activated: state=\(activationState.rawValue)")

        DispatchQueue.main.async { [weak self] in
            self?.isReachable = reachable
        }
    }

    nonisolated func sessionReachabilityDidChange(_ session: WCSession) {
        let reachable = session.isReachable
        Self.logger.info("iPhone reachability changed: \(reachable)")

        DispatchQueue.main.async { [weak self] in
            self?.isReachable = reachable
        }
    }

#if os(iOS)
    nonisolated func sessionDidBecomeInactive(_ session: WCSession) {
        Self.logger.info("WCSession became inactive")
    }

    nonisolated func sessionDidDeactivate(_ session: WCSession) {
        Self.logger.info("WCSession deactivated, reactivating")
        WCSession.default.activate()
    }
#endif

}
