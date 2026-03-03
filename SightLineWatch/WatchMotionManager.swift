//
//  WatchMotionManager.swift
//  SightLineWatch
//
//  Collects accelerometer and device-motion data on watchOS.
//  Requires an active HKWorkoutSession for background delivery.
//

import CoreMotion
import Foundation
import Combine
import os

/// Collects accelerometer and device-motion data on watchOS.
/// Requires an active HKWorkoutSession for background delivery.
class WatchMotionManager: ObservableObject {
    @Published var pitch: Double = 0.0
    @Published var roll: Double = 0.0
    @Published var yaw: Double = 0.0
    @Published var stabilityScore: Double = 1.0  // 1.0 = stable

    private static let logger = Logger(subsystem: "com.sightline.watch", category: "Motion")
    private let motionManager = CMMotionManager()
    private var recentAccelMagnitudes: [Double] = []
    private let stabilityWindowSize = 20  // ~2 seconds at 10 Hz

    func startUpdates() {
        guard motionManager.isDeviceMotionAvailable else {
            Self.logger.warning("Device motion unavailable")
            return
        }

        motionManager.deviceMotionUpdateInterval = 0.1  // 10 Hz (power-friendly)
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self, let motion else { return }

            self.pitch = motion.attitude.pitch
            self.roll = motion.attitude.roll
            self.yaw = motion.attitude.yaw

            // Compute stability score from acceleration variance
            let accel = motion.userAcceleration
            let magnitude = sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z)
            self.recentAccelMagnitudes.append(magnitude)
            if self.recentAccelMagnitudes.count > self.stabilityWindowSize {
                self.recentAccelMagnitudes.removeFirst()
            }
            self.stabilityScore = self.computeStability()
        }
        Self.logger.info("Device motion updates started at 10 Hz")
    }

    func stopUpdates() {
        motionManager.stopDeviceMotionUpdates()
        recentAccelMagnitudes.removeAll()
        Self.logger.info("Device motion updates stopped")
    }

    private func computeStability() -> Double {
        guard recentAccelMagnitudes.count >= 5 else { return 1.0 }
        let mean = recentAccelMagnitudes.reduce(0, +) / Double(recentAccelMagnitudes.count)
        let variance = recentAccelMagnitudes.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            / Double(recentAccelMagnitudes.count)
        // Map variance to 0-1: low variance = high stability
        // variance ~0 → score 1.0, variance > 0.5 → score ~0
        return max(0.0, min(1.0, 1.0 - (variance * 2.0)))
    }
}
