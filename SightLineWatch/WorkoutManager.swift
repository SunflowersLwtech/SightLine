//
//  WorkoutManager.swift
//  SightLineWatch
//
//  Manages HKWorkoutSession for continuous heart rate and sensor capture.
//  When a workout is active, Apple Watch samples heart rate every 1-5 seconds
//  (vs. 5-15 minutes at rest). Each new reading is forwarded to iPhone
//  via PhoneConnector (WCSession) along with motion, heading, and health context.
//
//  Reference: Apple SpeedySloth sample, extended for full sensor payload.
//

import HealthKit
import Foundation
import Combine
import os

class WorkoutManager: NSObject, ObservableObject {
    // MARK: - Published State

    @Published var heartRate: Double = 0
    @Published var isRunning: Bool = false
    @Published var isAuthorized: Bool = false

    // MARK: - Sensor Managers

    let watchMotion = WatchMotionManager()
    let watchHeading = WatchHeadingManager()
    let watchHealth = WatchHealthContext()

    // MARK: - Private

    nonisolated private static let logger = Logger(
        subsystem: "com.sightline.watch",
        category: "Workout"
    )

    private let healthStore = HKHealthStore()
    private var session: HKWorkoutSession?
    private var builder: HKLiveWorkoutBuilder?

    // MARK: - Authorization

    /// Request HealthKit read permission for heart rate.
    func requestAuthorization() async {
        let heartRateType = HKQuantityType(.heartRate)

        do {
            try await healthStore.requestAuthorization(toShare: [.workoutType()], read: [heartRateType])
            await MainActor.run {
                isAuthorized = true
            }
            Self.logger.info("HealthKit authorization granted")
        } catch {
            Self.logger.error("HealthKit authorization failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Workout Lifecycle

    /// Start a workout session to enable high-frequency heart rate sampling.
    func startWorkout() {
        guard !isRunning else { return }

        Task {
            if !isAuthorized {
                await requestAuthorization()
            }

            await _startWorkoutSession()
        }
    }

    /// Stop the active workout session.
    func stopWorkout() {
        guard isRunning else { return }

        session?.end()

        builder?.endCollection(withEnd: Date()) { [weak self] success, error in
            if let error = error {
                Self.logger.error("End collection failed: \(error.localizedDescription)")
            }

            self?.builder?.finishWorkout { workout, error in
                if let error = error {
                    Self.logger.error("Finish workout failed: \(error.localizedDescription)")
                }
                Self.logger.info("Workout finished successfully")
            }
        }

        isRunning = false

        // Stop sensor managers
        watchMotion.stopUpdates()
        watchHeading.stopUpdates()
        watchHealth.stop()

        // Notify iPhone that monitoring stopped
        PhoneConnector.shared.sendHeartRate(0, isMonitoring: false)

        Self.logger.info("Workout stopped")
    }

    // MARK: - Private

    private func _startWorkoutSession() async {
        let config = HKWorkoutConfiguration()
        config.activityType = .other          // Generic activity — just need HR
        config.locationType = .outdoor        // Outdoor enables GPS if needed

        do {
            session = try HKWorkoutSession(
                healthStore: healthStore,
                configuration: config
            )
            builder = session?.associatedWorkoutBuilder()

            session?.delegate = self
            builder?.delegate = self

            builder?.dataSource = HKLiveWorkoutDataSource(
                healthStore: healthStore,
                workoutConfiguration: config
            )

            let startDate = Date()
            session?.startActivity(with: startDate)

            try await builder?.beginCollection(at: startDate)

            await MainActor.run {
                isRunning = true
                watchMotion.startUpdates()
                watchHeading.startUpdates()
            }
            await watchHealth.requestAuthorizationAndStart()

            Self.logger.info("Workout session started — high-frequency HR + sensors active")
        } catch {
            Self.logger.error("Failed to start workout: \(error.localizedDescription)")
        }
    }

    /// Extract latest heart rate from workout builder statistics.
    nonisolated private func latestHeartRate(from builder: HKLiveWorkoutBuilder) -> Double? {
        let heartRateType = HKQuantityType(.heartRate)
        guard let statistics = builder.statistics(for: heartRateType),
              let quantity = statistics.mostRecentQuantity() else {
            return nil
        }

        let bpm = quantity.doubleValue(
            for: HKUnit.count().unitDivided(by: .minute())
        )
        return bpm > 0 ? bpm : nil
    }
}

// MARK: - HKWorkoutSessionDelegate

extension WorkoutManager: HKWorkoutSessionDelegate {
    nonisolated func workoutSession(
        _ workoutSession: HKWorkoutSession,
        didChangeTo toState: HKWorkoutSessionState,
        from fromState: HKWorkoutSessionState,
        date: Date
    ) {
        Self.logger.info("Workout state: \(fromState.rawValue) → \(toState.rawValue)")

        if toState == .ended {
            DispatchQueue.main.async { [weak self] in
                self?.isRunning = false
            }
        }
    }

    nonisolated func workoutSession(
        _ workoutSession: HKWorkoutSession,
        didFailWithError error: Error
    ) {
        Self.logger.error("Workout session failed: \(error.localizedDescription)")

        DispatchQueue.main.async { [weak self] in
            self?.isRunning = false
        }
    }
}

// MARK: - HKLiveWorkoutBuilderDelegate

extension WorkoutManager: HKLiveWorkoutBuilderDelegate {
    nonisolated func workoutBuilder(
        _ workoutBuilder: HKLiveWorkoutBuilder,
        didCollectDataOf collectedTypes: Set<HKSampleType>
    ) {
        let hasHeartRateData = collectedTypes.contains { type in
            guard let quantityType = type as? HKQuantityType else { return false }
            return quantityType == HKQuantityType(.heartRate)
        }

        guard hasHeartRateData,
              let bpm = latestHeartRate(from: workoutBuilder) else {
            return
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.heartRate = bpm
            PhoneConnector.shared.sendWatchContext(
                heartRate: bpm,
                isMonitoring: true,
                motion: (
                    pitch: self.watchMotion.pitch,
                    roll: self.watchMotion.roll,
                    yaw: self.watchMotion.yaw,
                    stabilityScore: self.watchMotion.stabilityScore
                ),
                heading: self.watchHeading.heading,
                headingAccuracy: self.watchHeading.headingAccuracy,
                spO2: self.watchHealth.spO2,
                noiseExposure: self.watchHealth.noiseExposure
            )
        }

        Self.logger.debug("Heart rate: \(Int(bpm)) BPM + context → sent to iPhone")
    }

    nonisolated func workoutBuilderDidCollectEvent(
        _ workoutBuilder: HKLiveWorkoutBuilder
    ) {
        // Not used — we only care about heart rate data
    }
}
