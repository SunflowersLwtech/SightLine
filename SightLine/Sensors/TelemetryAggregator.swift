//
//  TelemetryAggregator.swift
//  SightLine
//
//  LOD-aware telemetry throttler. Adjusts send frequency based on
//  the current LOD level and detects immediate trigger conditions
//  that bypass throttling (motion state changes, HR spikes, etc.).
//

import Foundation
import Combine
import os

class TelemetryAggregator: ObservableObject {
    @Published var currentLOD: Int = 2
    @Published var isPaused: Bool = false

    private static let logger = Logger(subsystem: "com.sightline.app", category: "Telemetry")
    // SL-31 telemetry interval windows:
    // LOD 1: 3-4s, LOD 2: 2-3s, LOD 3: 5-10s.
    private let lodTelemetryIntervalWindow: [Int: (TimeInterval, TimeInterval)] = [
        1: (3.0, 4.0),
        2: (2.0, 3.0),
        3: (5.0, 10.0)
    ]

    private var timer: Timer?
    private weak var sensorManager: SensorManager?
    private weak var webSocketManager: WebSocketManager?
    private var lastSentTelemetry: TelemetryData?
    private var lastSendTime: Date = .distantPast

    /// Conditions that trigger immediate telemetry send, bypassing throttle.
    enum ImmediateTrigger: String {
        case motionStateChanged
        case heartRateSpike
        case noiseThresholdCrossed
        case userGesture
    }

    /// LOD-aware send interval in seconds.
    var sendInterval: TimeInterval {
        let window = lodTelemetryIntervalWindow[currentLOD] ?? lodTelemetryIntervalWindow[2]!
        return (window.0 + window.1) / 2.0
    }

    /// Start the telemetry aggregation loop.
    func start(sensorManager: SensorManager, webSocket: WebSocketManager) {
        self.sensorManager = sensorManager
        self.webSocketManager = webSocket
        isPaused = false

        // Schedule repeating timer on main run loop
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.tick()
        }

        Self.logger.info("TelemetryAggregator started (LOD \(self.currentLOD), interval \(self.sendInterval)s)")
    }

    /// Stop the telemetry aggregation loop.
    func stop() {
        timer?.invalidate()
        timer = nil
        Self.logger.info("TelemetryAggregator stopped")
    }

    /// Pause sending (e.g. during disconnection). Sensors keep collecting.
    func pause() {
        isPaused = true
        Self.logger.info("TelemetryAggregator paused")
    }

    /// Resume sending after reconnection.
    func resume() {
        isPaused = false
        Self.logger.info("TelemetryAggregator resumed")
    }

    /// Update LOD level (called when backend sends lodUpdate).
    func updateLOD(_ lod: Int) {
        let oldLOD = currentLOD
        currentLOD = lod
        if oldLOD != lod {
            Self.logger.info("LOD updated: \(oldLOD) → \(lod)")
        }
    }

    /// Send a gesture event immediately.
    func sendGesture(_ gesture: String) {
        guard let sensor = sensorManager, let ws = webSocketManager else { return }
        var data = sensor.snapshot()
        data.userGesture = gesture
        sendTelemetry(data, via: ws, trigger: .userGesture)
    }

    // MARK: - Private

    private func tick() {
        guard !isPaused else { return }
        guard let sensor = sensorManager, let ws = webSocketManager else { return }

        let newData = sensor.snapshot()

        // Check for immediate triggers (bypass throttle)
        if let trigger = immediateTrigger(old: lastSentTelemetry, new: newData) {
            sendTelemetry(newData, via: ws, trigger: trigger)
            return
        }

        // Scheduled send based on LOD interval
        if Date().timeIntervalSince(lastSendTime) >= sendInterval {
            sendTelemetry(newData, via: ws, trigger: nil)
        }
    }

    private func sendTelemetry(_ data: TelemetryData, via ws: WebSocketManager, trigger: ImmediateTrigger?) {
        let msg = UpstreamMessage.telemetry(data: data)
        ws.sendText(msg.toJSON())
        lastSentTelemetry = data
        lastSendTime = Date()

        if let trigger = trigger {
            Self.logger.info("Immediate telemetry sent: \(trigger.rawValue)")
        }
    }

    /// Evaluates whether telemetry should bypass throttle and send immediately.
    /// Kept internal so unit tests can verify trigger rules.
    func immediateTrigger(old: TelemetryData?, new: TelemetryData) -> ImmediateTrigger? {
        guard let old = old else { return nil }

        // Motion state changed
        if old.motionState != new.motionState {
            return .motionStateChanged
        }

        // Heart rate spike (>30% change or >120 BPM)
        if let newHR = new.heartRate, let oldHR = old.heartRate {
            if newHR > 120 {
                return .heartRateSpike
            }
            if oldHR > 0 && abs(newHR - oldHR) / oldHR > 0.3 {
                return .heartRateSpike
            }
        }

        // Noise threshold crossed (40dB or 80dB boundaries)
        let oldNoise = old.ambientNoiseDb
        let newNoise = new.ambientNoiseDb
        let crossedLow = (oldNoise >= 40 && newNoise < 40) || (oldNoise < 40 && newNoise >= 40)
        let crossedHigh = (oldNoise >= 80 && newNoise < 80) || (oldNoise < 80 && newNoise >= 80)
        if crossedLow || crossedHigh {
            return .noiseThresholdCrossed
        }

        // User gesture
        if new.userGesture != nil && old.userGesture == nil {
            return .userGesture
        }

        return nil
    }
}
