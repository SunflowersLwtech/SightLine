//
//  SensorManagerTests.swift
//  SightLineTests
//
//  Smoke tests for sensor data collection and telemetry.
//

import Testing
import Foundation
@testable import SightLine

@Suite("Sensor Manager Tests")
struct SensorManagerTests {

    @Test("SensorManager initializes")
    func initialization() {
        let manager = SensorManager()
        #expect(manager != nil)
    }

    @Test("TelemetryData default values are sensible")
    func telemetryDefaults() {
        let telemetry = TelemetryData()
        #expect(telemetry.motionState == "stationary")
        #expect(telemetry.stepCadence == 0)
        #expect(telemetry.ambientNoiseDb == 50.0)
        #expect(telemetry.deviceType == "phone_only")
    }

    @Test("TelemetryData encodes with snake_case keys")
    func telemetryEncoding() throws {
        var telemetry = TelemetryData()
        telemetry.motionState = "walking"
        telemetry.heartRate = 72.0

        let data = try JSONEncoder().encode(telemetry)
        let json = String(data: data, encoding: .utf8)!

        #expect(json.contains("motion_state"))
        #expect(json.contains("heart_rate"))
    }

    @Test("SensorManager start and stop do not crash")
    func startStop() {
        let manager = SensorManager()
        manager.startAll()
        manager.stopAll()
    }
}
