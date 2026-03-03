//
//  MotionManager.swift
//  SightLine
//
//  Provides motion state and step cadence using CMMotionActivityManager
//  and CMPedometer. Feeds TelemetryAggregator for LOD-aware decisions.
//

import CoreMotion
import Foundation
import Combine
import os

class MotionManager: ObservableObject {
    @Published var motionState: String = "stationary"
    @Published var stepCadence: Double = 0.0

    private static let logger = Logger(subsystem: "com.sightline.app", category: "Motion")

    private let activityManager = CMMotionActivityManager()
    private let pedometer = CMPedometer()
    private var isMonitoring = false

    /// Start motion and pedometer monitoring.
    func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true
        startActivityUpdates()
        startPedometerUpdates()
    }

    /// Stop all motion monitoring.
    func stopMonitoring() {
        guard isMonitoring else { return }
        isMonitoring = false
        activityManager.stopActivityUpdates()
        pedometer.stopUpdates()
        Self.logger.info("Motion monitoring stopped")
    }

    // MARK: - Activity Updates

    private func startActivityUpdates() {
        guard CMMotionActivityManager.isActivityAvailable() else {
            Self.logger.warning("Motion activity not available on this device")
            return
        }

        activityManager.startActivityUpdates(to: .main) { [weak self] activity in
            guard let self = self, let activity = activity else { return }
            let state = self.mapActivityToState(activity)
            if state != self.motionState {
                Self.logger.info("Motion state changed: \(self.motionState) → \(state)")
                self.motionState = state
            }
        }
        Self.logger.info("Activity updates started")
    }

    private func mapActivityToState(_ activity: CMMotionActivity) -> String {
        if activity.running { return "running" }
        if activity.cycling { return "cycling" }
        if activity.automotive { return "automotive" }
        if activity.walking { return "walking" }
        if activity.stationary { return "stationary" }
        return "stationary"
    }

    // MARK: - Pedometer Updates (Step Cadence)

    private func startPedometerUpdates() {
        guard CMPedometer.isStepCountingAvailable() else {
            Self.logger.warning("Step counting not available")
            return
        }

        pedometer.startUpdates(from: Date()) { [weak self] data, error in
            guard let self = self else { return }
            if let error = error {
                Self.logger.error("Pedometer error: \(error.localizedDescription)")
                return
            }
            guard let data = data else { return }

            // Calculate steps/min from currentCadence if available (iOS 9+)
            // currentCadence is steps per second
            if let cadence = data.currentCadence {
                let stepsPerMin = cadence.doubleValue * 60.0
                DispatchQueue.main.async {
                    self.stepCadence = stepsPerMin
                }
            }
        }
        Self.logger.info("Pedometer updates started")
    }
}
