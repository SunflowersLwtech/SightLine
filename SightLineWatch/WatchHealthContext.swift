//
//  WatchHealthContext.swift
//  SightLineWatch
//
//  Reads SpO2 and environmental noise exposure from HealthKit on watch.
//  These are system-measured values — we cannot trigger measurements.
//

import HealthKit
import Foundation
import Combine
import os

/// Reads SpO2 and environmental noise exposure from HealthKit on watch.
/// These are system-measured values — we cannot trigger measurements.
class WatchHealthContext: ObservableObject {
    @Published var spO2: Double?
    @Published var noiseExposure: Double?

    private static let logger = Logger(subsystem: "com.sightline.watch", category: "HealthContext")
    private let healthStore = HKHealthStore()
    private var spO2Query: HKAnchoredObjectQuery?
    private var noiseQuery: HKAnchoredObjectQuery?

    func requestAuthorizationAndStart() async {
        let types: Set<HKSampleType> = [
            HKQuantityType(.oxygenSaturation),
            HKQuantityType(.environmentalAudioExposure),
        ]

        do {
            try await healthStore.requestAuthorization(toShare: [], read: types)
            startObserving()
            Self.logger.info("HealthContext authorized and observing")
        } catch {
            Self.logger.error("HealthContext authorization failed: \(error.localizedDescription)")
        }
    }

    private func startObserving() {
        // SpO2
        let spO2Type = HKQuantityType(.oxygenSaturation)
        spO2Query = HKAnchoredObjectQuery(
            type: spO2Type,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            self?.processSpO2(samples)
        }
        spO2Query?.updateHandler = { [weak self] _, samples, _, _, _ in
            self?.processSpO2(samples)
        }
        if let q = spO2Query { healthStore.execute(q) }

        // Environmental noise
        let noiseType = HKQuantityType(.environmentalAudioExposure)
        noiseQuery = HKAnchoredObjectQuery(
            type: noiseType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            self?.processNoise(samples)
        }
        noiseQuery?.updateHandler = { [weak self] _, samples, _, _, _ in
            self?.processNoise(samples)
        }
        if let q = noiseQuery { healthStore.execute(q) }
    }

    private func processSpO2(_ samples: [HKSample]?) {
        guard let latest = samples?.compactMap({ $0 as? HKQuantitySample }).last else { return }
        let value = latest.quantity.doubleValue(for: .percent()) * 100.0
        DispatchQueue.main.async { self.spO2 = value }
        Self.logger.debug("SpO2 updated: \(value)%")
    }

    private func processNoise(_ samples: [HKSample]?) {
        guard let latest = samples?.compactMap({ $0 as? HKQuantitySample }).last else { return }
        let value = latest.quantity.doubleValue(for: .decibelAWeightedSoundPressureLevel())
        DispatchQueue.main.async { self.noiseExposure = value }
        Self.logger.debug("Noise exposure updated: \(value) dB")
    }

    func stop() {
        if let q = spO2Query { healthStore.stop(q) }
        if let q = noiseQuery { healthStore.stop(q) }
    }
}
