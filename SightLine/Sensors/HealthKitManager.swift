//
//  HealthKitManager.swift
//  SightLine
//
//  Reads heart rate data from HealthKit as a backup channel.
//  Primary real-time heart rate comes from WCSession (Phase 3 watchOS).
//  This HealthKit path has 10-20 minute system sync delay from Apple Watch.
//  Gracefully degrades if HealthKit is unavailable or permission denied.
//

import HealthKit
import Foundation
import Combine
import os

class HealthKitManager: ObservableObject {
    @Published var heartRate: Double? = nil
    @Published var isAvailable: Bool = false

    private static let logger = Logger(subsystem: "com.sightline.app", category: "HealthKit")

    private var healthStore: HKHealthStore?
    private var heartRateQuery: HKAnchoredObjectQuery?

    init() {
        if HKHealthStore.isHealthDataAvailable() {
            healthStore = HKHealthStore()
            isAvailable = true
        } else {
            Self.logger.info("HealthKit not available on this device")
        }
    }

    /// Request read permission for heart rate data.
    func requestAuthorization() async {
        guard let healthStore = healthStore else { return }

        let heartRateType = HKQuantityType(.heartRate)

        do {
            try await healthStore.requestAuthorization(toShare: [], read: [heartRateType])
            Self.logger.info("HealthKit authorization granted")
        } catch {
            Self.logger.error("HealthKit authorization failed: \(error.localizedDescription)")
        }
    }

    /// Start monitoring heart rate via HKAnchoredObjectQuery.
    func startMonitoring() {
        guard let healthStore = healthStore else { return }

        let heartRateType = HKQuantityType(.heartRate)

        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, error in
            if let error = error {
                Self.logger.error("HealthKit query error: \(error.localizedDescription)")
                return
            }
            self?.processHeartRateSamples(samples)
        }

        query.updateHandler = { [weak self] _, samples, _, _, error in
            if let error = error {
                Self.logger.error("HealthKit update error: \(error.localizedDescription)")
                return
            }
            self?.processHeartRateSamples(samples)
        }

        healthStore.execute(query)
        heartRateQuery = query
        Self.logger.info("HealthKit heart rate monitoring started")
    }

    /// Stop heart rate monitoring.
    func stopMonitoring() {
        if let query = heartRateQuery, let healthStore = healthStore {
            healthStore.stop(query)
            heartRateQuery = nil
        }
        Self.logger.info("HealthKit monitoring stopped")
    }

    private func processHeartRateSamples(_ samples: [HKSample]?) {
        guard let quantitySamples = samples as? [HKQuantitySample],
              let latest = quantitySamples.last else { return }

        let bpm = latest.quantity.doubleValue(
            for: HKUnit.count().unitDivided(by: .minute())
        )

        DispatchQueue.main.async {
            self.heartRate = bpm
        }
    }
}
