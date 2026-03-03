//
//  TelemetryAggregatorTests.swift
//  SightLineTests
//
//  Validates immediate trigger rules for telemetry sending.
//

import Testing
import Foundation
@testable import SightLine

@Suite("Telemetry Aggregator")
struct TelemetryAggregatorTests {

    @Test("LOD send interval follows midpoint windows")
    func sendIntervalByLod() {
        let aggregator = TelemetryAggregator()

        aggregator.updateLOD(1)
        #expect(aggregator.sendInterval == 3.5)

        aggregator.updateLOD(2)
        #expect(aggregator.sendInterval == 2.5)

        aggregator.updateLOD(3)
        #expect(aggregator.sendInterval == 7.5)
    }

    @Test("motion state change triggers immediate send")
    func motionStateChangeTrigger() {
        let aggregator = TelemetryAggregator()
        let old = makeTelemetry(motionState: "walking")
        let new = makeTelemetry(motionState: "running")

        #expect(aggregator.immediateTrigger(old: old, new: new) == .motionStateChanged)
    }

    @Test("heart rate above 120 triggers immediate send")
    func heartRateCeilingTrigger() {
        let aggregator = TelemetryAggregator()
        let old = makeTelemetry(heartRate: 90)
        let new = makeTelemetry(heartRate: 121)

        #expect(aggregator.immediateTrigger(old: old, new: new) == .heartRateSpike)
    }

    @Test("heart rate spike over 30 percent triggers immediate send")
    func heartRateSpikeTrigger() {
        let aggregator = TelemetryAggregator()
        let old = makeTelemetry(heartRate: 80)
        let new = makeTelemetry(heartRate: 110)

        #expect(aggregator.immediateTrigger(old: old, new: new) == .heartRateSpike)
    }

    @Test("noise threshold crossing triggers immediate send")
    func noiseThresholdTrigger() {
        let aggregator = TelemetryAggregator()
        let old = makeTelemetry(ambientNoiseDb: 45)
        let new = makeTelemetry(ambientNoiseDb: 35)

        #expect(aggregator.immediateTrigger(old: old, new: new) == .noiseThresholdCrossed)
    }

    @Test("user gesture triggers immediate send")
    func userGestureTrigger() {
        let aggregator = TelemetryAggregator()
        let old = makeTelemetry(userGesture: nil)
        let new = makeTelemetry(userGesture: "lod_up")

        #expect(aggregator.immediateTrigger(old: old, new: new) == .userGesture)
    }

    @Test("no significant change does not trigger immediate send")
    func noTriggerWhenStable() {
        let aggregator = TelemetryAggregator()
        let old = makeTelemetry(
            motionState: "walking",
            ambientNoiseDb: 60,
            heartRate: 88
        )
        let new = makeTelemetry(
            motionState: "walking",
            ambientNoiseDb: 62,
            heartRate: 90
        )

        #expect(aggregator.immediateTrigger(old: old, new: new) == nil)
    }

    private func makeTelemetry(
        motionState: String = "stationary",
        ambientNoiseDb: Double = 50,
        heartRate: Double? = nil,
        userGesture: String? = nil
    ) -> TelemetryData {
        var telemetry = TelemetryData()
        telemetry.motionState = motionState
        telemetry.ambientNoiseDb = ambientNoiseDb
        telemetry.heartRate = heartRate
        telemetry.userGesture = userGesture
        return telemetry
    }
}
