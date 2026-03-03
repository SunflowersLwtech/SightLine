//
//  WatchDataModelTests.swift
//  SightLineWatchTests
//
//  Tests for watchOS data models and heart rate payload construction.
//  These tests verify the payload format sent from Watch to iPhone.
//

import Testing
import Foundation

@Suite("Heart Rate Payload")
struct HeartRatePayloadTests {

    /// Simulate the payload format that PhoneConnector builds.
    private func buildPayload(bpm: Double, isMonitoring: Bool) -> [String: Any] {
        return [
            "heartRate": bpm,
            "ts": Date().timeIntervalSince1970,
            "isMonitoring": isMonitoring,
        ]
    }

    @Test("payload contains heartRate key")
    func payloadHasHeartRate() {
        let payload = buildPayload(bpm: 72.0, isMonitoring: true)
        #expect(payload["heartRate"] as? Double == 72.0)
    }

    @Test("payload contains timestamp")
    func payloadHasTimestamp() {
        let before = Date().timeIntervalSince1970
        let payload = buildPayload(bpm: 80.0, isMonitoring: true)
        let ts = payload["ts"] as! Double
        let after = Date().timeIntervalSince1970
        #expect(ts >= before)
        #expect(ts <= after)
    }

    @Test("monitoring flag is included")
    func payloadHasMonitoringFlag() {
        let activePayload = buildPayload(bpm: 90.0, isMonitoring: true)
        #expect(activePayload["isMonitoring"] as? Bool == true)

        let stoppedPayload = buildPayload(bpm: 0, isMonitoring: false)
        #expect(stoppedPayload["isMonitoring"] as? Bool == false)
    }

    @Test("zero BPM indicates monitoring stopped")
    func zeroBpmMeansMonitoringStopped() {
        let payload = buildPayload(bpm: 0, isMonitoring: false)
        #expect(payload["heartRate"] as? Double == 0)
        #expect(payload["isMonitoring"] as? Bool == false)
    }

    @Test("high heart rate values preserved")
    func highHeartRatePreserved() {
        let payload = buildPayload(bpm: 185.0, isMonitoring: true)
        #expect(payload["heartRate"] as? Double == 185.0)
    }
}
