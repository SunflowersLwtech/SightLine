//
//  MessageProtocolTests.swift
//  SightLineTests
//
//  Tests for the WebSocket message protocol: upstream encoding, downstream parsing,
//  telemetry data serialization, and tool behavior mode extraction.
//

import Testing
import Foundation
@testable import SightLine

@Suite("UpstreamMessage Encoding")
struct UpstreamMessageTests {

    @Test("audio message encodes to JSON with base64 data")
    func audioEncoding() {
        let data = Data([0x01, 0x02, 0x03])
        let msg = UpstreamMessage.audio(data: data)
        let json = msg.toJSON()

        #expect(json.contains("\"type\":\"audio\""))
        #expect(json.contains("\"data\":\""))
        #expect(json.contains(data.base64EncodedString()))
    }

    @Test("image message includes mimeType")
    func imageEncoding() {
        let data = Data([0xFF, 0xD8])
        let msg = UpstreamMessage.image(data: data, mimeType: "image/jpeg")
        let json = msg.toJSON()

        #expect(json.contains("\"type\":\"image\""))
        #expect(json.contains("\"mimeType\":\"image/jpeg\""))
        #expect(json.contains(data.base64EncodedString()))
    }

    @Test("telemetry message encodes TelemetryData")
    func telemetryEncoding() throws {
        var telemetry = TelemetryData()
        telemetry.motionState = "walking"
        telemetry.stepCadence = 110
        telemetry.ambientNoiseDb = 65.5
        telemetry.heartRate = 80.0

        let msg = UpstreamMessage.telemetry(data: telemetry)
        let json = msg.toJSON()

        #expect(json.contains("\"type\":\"telemetry\""))
        #expect(json.contains("\"motion_state\":\"walking\""))
        #expect(json.contains("\"step_cadence\":110"))
    }

    @Test("reloadFaceLibrary encodes correctly")
    func reloadFaceLibraryEncoding() {
        let msg = UpstreamMessage.reloadFaceLibrary
        #expect(msg.toJSON() == "{\"type\":\"reload_face_library\"}")
    }

    @Test("clearFaceLibrary encodes correctly")
    func clearFaceLibraryEncoding() {
        let msg = UpstreamMessage.clearFaceLibrary
        #expect(msg.toJSON() == "{\"type\":\"clear_face_library\"}")
    }

    @Test("cameraFailure encodes error and reason")
    func cameraFailureEncoding() {
        let msg = UpstreamMessage.cameraFailure(error: "no permission")
        let json = msg.toJSON()
        #expect(json.contains("\"type\":\"camera_failure\""))
        #expect(json.contains("\"error\":\"no permission\""))
        #expect(json.contains("\"reason\":\"no permission\""))
    }

    @Test("cameraFailure escapes special characters")
    func cameraFailureEscaping() {
        let msg = UpstreamMessage.cameraFailure(error: "bad \"quote\" and \\slash")
        let json = msg.toJSON()
        #expect(json.contains("\\\"quote\\\""))
        #expect(json.contains("\\\\slash"))
    }

    @Test("muteToggle encodes muted state")
    func muteToggleEncoding() {
        let muted = UpstreamMessage.muteToggle(muted: true)
        #expect(muted.toJSON() == "{\"type\":\"gesture\",\"gesture\":\"mute_toggle\",\"muted\":true}")

        let unmuted = UpstreamMessage.muteToggle(muted: false)
        #expect(unmuted.toJSON() == "{\"type\":\"gesture\",\"gesture\":\"mute_toggle\",\"muted\":false}")
    }

    @Test("cameraToggle encodes active state")
    func cameraToggleEncoding() {
        let on = UpstreamMessage.cameraToggle(active: true)
        #expect(on.toJSON() == "{\"type\":\"gesture\",\"gesture\":\"camera_toggle\",\"active\":true}")

        let off = UpstreamMessage.cameraToggle(active: false)
        #expect(off.toJSON() == "{\"type\":\"gesture\",\"gesture\":\"camera_toggle\",\"active\":false}")
    }
}

@Suite("DownstreamMessage Parsing")
struct DownstreamMessageTests {

    @Test("transcript message parses text and role")
    func transcriptParsing() {
        let json = "{\"type\":\"transcript\",\"text\":\"Hello world\",\"role\":\"agent\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .transcript(let text, let role) = msg {
            #expect(text == "Hello world")
            #expect(role == "agent")
        } else {
            Issue.record("Expected transcript message")
        }
    }

    @Test("lodUpdate parses LOD level")
    func lodUpdateParsing() {
        let json = "{\"type\":\"lod_update\",\"lod\":3}"
        let msg = DownstreamMessage.parse(text: json)

        if case .lodUpdate(let lod) = msg {
            #expect(lod == 3)
        } else {
            Issue.record("Expected lodUpdate message")
        }
    }

    @Test("goAway parses retry_ms")
    func goAwayParsing() {
        let json = "{\"type\":\"go_away\",\"retry_ms\":2000}"
        let msg = DownstreamMessage.parse(text: json)

        if case .goAway(let retryMs) = msg {
            #expect(retryMs == 2000)
        } else {
            Issue.record("Expected goAway message")
        }
    }

    @Test("sessionResumption parses handle")
    func sessionResumptionParsing() {
        let json = "{\"type\":\"session_resumption\",\"handle\":\"abc-123\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .sessionResumption(let handle) = msg {
            #expect(handle == "abc-123")
        } else {
            Issue.record("Expected sessionResumption message")
        }
    }

    @Test("session_ready parses")
    func sessionReadyParsing() {
        let json = "{\"type\":\"session_ready\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .sessionReady = msg {
            // pass
        } else {
            Issue.record("Expected sessionReady message")
        }
    }

    @Test("face_library_reloaded parses count")
    func faceLibraryReloadedParsing() {
        let json = "{\"type\":\"face_library_reloaded\",\"count\":4}"
        let msg = DownstreamMessage.parse(text: json)

        if case .faceLibraryReloaded(let count) = msg {
            #expect(count == 4)
        } else {
            Issue.record("Expected faceLibraryReloaded message")
        }
    }

    @Test("face_library_cleared parses deleted_count")
    func faceLibraryClearedParsing() {
        let json = "{\"type\":\"face_library_cleared\",\"deleted_count\":2}"
        let msg = DownstreamMessage.parse(text: json)

        if case .faceLibraryCleared(let deletedCount) = msg {
            #expect(deletedCount == 2)
        } else {
            Issue.record("Expected faceLibraryCleared message")
        }
    }

    @Test("error parses message")
    func errorParsing() {
        let json = "{\"type\":\"error\",\"error\":\"Face recognition unavailable\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .error(let message) = msg {
            #expect(message == "Face recognition unavailable")
        } else {
            Issue.record("Expected error message")
        }
    }

    @Test("vision_result parses summary and behavior")
    func visionResultParsing() {
        let json = "{\"type\":\"vision_result\",\"summary\":\"Crosswalk ahead\",\"behavior\":\"INTERRUPT\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .visionResult(let summary, let behavior) = msg {
            #expect(summary == "Crosswalk ahead")
            #expect(behavior == .INTERRUPT)
        } else {
            Issue.record("Expected visionResult message")
        }
    }

    @Test("ocr_result parses summary from nested data")
    func ocrResultFromData() {
        let json = "{\"type\":\"ocr_result\",\"data\":{\"summary\":\"Menu: Coffee $3\"},\"behavior\":\"WHEN_IDLE\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .ocrResult(let summary, let behavior) = msg {
            #expect(summary == "Menu: Coffee $3")
            #expect(behavior == .WHEN_IDLE)
        } else {
            Issue.record("Expected ocrResult message")
        }
    }

    @Test("vision_debug parses data payload")
    func visionDebugParsing() {
        let json = "{\"type\":\"vision_debug\",\"data\":{\"bounding_boxes\":[{\"box_2d\":[100,200,500,700],\"label\":\"stairs\"}]}}"
        let msg = DownstreamMessage.parse(text: json)

        if case .visionDebug(let data) = msg {
            let boxes = data["bounding_boxes"] as? [[String: Any]]
            #expect(boxes?.count == 1)
        } else {
            Issue.record("Expected visionDebug message")
        }
    }

    @Test("frame_ack parses frame id and queued agents")
    func frameAckParsing() {
        let json = "{\"type\":\"frame_ack\",\"frame_id\":42,\"queued_agents\":[\"vision\",\"ocr\"]}"
        let msg = DownstreamMessage.parse(text: json)

        if case .frameAck(let frameId, let queuedAgents) = msg {
            #expect(frameId == 42)
            #expect(queuedAgents == ["vision", "ocr"])
        } else {
            Issue.record("Expected frameAck message")
        }
    }

    @Test("navigation_result parses")
    func navigationResultParsing() {
        let json = "{\"type\":\"navigation_result\",\"summary\":\"Turn left in 50 meters\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .navigationResult(let summary, _) = msg {
            #expect(summary == "Turn left in 50 meters")
        } else {
            Issue.record("Expected navigationResult message")
        }
    }

    @Test("person_identified parses name")
    func personIdentifiedParsing() {
        let json = "{\"type\":\"person_identified\",\"person_name\":\"Alice\",\"behavior\":\"INTERRUPT\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .personIdentified(let name, let behavior) = msg {
            #expect(name == "Alice")
            #expect(behavior == .INTERRUPT)
        } else {
            Issue.record("Expected personIdentified message")
        }
    }

    @Test("identity_update parses matched flag")
    func identityUpdateParsing() {
        let json = "{\"type\":\"identity_update\",\"name\":\"Bob\",\"matched\":true,\"behavior\":\"SILENT\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .identityUpdate(let name, let matched, let behavior) = msg {
            #expect(name == "Bob")
            #expect(matched == true)
            #expect(behavior == .SILENT)
        } else {
            Issue.record("Expected identityUpdate message")
        }
    }

    @Test("tool_event parses tool name and payload")
    func toolEventParsing() {
        let json = "{\"type\":\"tool_event\",\"tool\":\"navigate_to\",\"data\":{\"destination\":\"Central Park\"}}"
        let msg = DownstreamMessage.parse(text: json)

        if case .toolEvent(let tool, _, let payload) = msg {
            #expect(tool == "navigate_to")
            #expect(payload["destination"] as? String == "Central Park")
        } else {
            Issue.record("Expected toolEvent message")
        }
    }

    @Test("unknown type returns .unknown")
    func unknownTypeParsing() {
        let json = "{\"type\":\"future_msg_type\",\"value\":42}"
        let msg = DownstreamMessage.parse(text: json)

        if case .unknown(let raw) = msg {
            #expect(raw == json)
        } else {
            Issue.record("Expected unknown message")
        }
    }

    @Test("invalid JSON returns .unknown")
    func invalidJsonParsing() {
        let msg = DownstreamMessage.parse(text: "not json at all")
        if case .unknown = msg {
            // pass
        } else {
            Issue.record("Expected unknown for invalid JSON")
        }
    }

    @Test("lodUpdate defaults to 2 when lod missing")
    func lodUpdateDefaultValue() {
        let json = "{\"type\":\"lod_update\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .lodUpdate(let lod) = msg {
            #expect(lod == 2)
        } else {
            Issue.record("Expected lodUpdate with default value")
        }
    }

    @Test("capability_degraded parses capability and reason")
    func capabilityDegradedParsing() {
        let json = "{\"type\":\"capability_degraded\",\"capability\":\"vision\",\"reason\":\"model timeout\",\"recoverable\":true}"
        let msg = DownstreamMessage.parse(text: json)

        if case .capabilityDegraded(let capability, let reason, let recoverable) = msg {
            #expect(capability == "vision")
            #expect(reason == "model timeout")
            #expect(recoverable == true)
        } else {
            Issue.record("Expected capabilityDegraded message")
        }
    }

    @Test("capability_degraded defaults recoverable to true")
    func capabilityDegradedDefaultRecoverable() {
        let json = "{\"type\":\"capability_degraded\",\"capability\":\"ocr\",\"reason\":\"error\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .capabilityDegraded(_, _, let recoverable) = msg {
            #expect(recoverable == true)
        } else {
            Issue.record("Expected capabilityDegraded message")
        }
    }

    @Test("debug_lod parses LOD debug data")
    func debugLodParsing() {
        let json = "{\"type\":\"debug_lod\",\"data\":{\"lod\":2,\"prev\":1,\"reason\":\"Rule2:stationary→LOD3\",\"rules\":[\"Rule2\"]}}"
        let msg = DownstreamMessage.parse(text: json)

        if case .debugLod(let data) = msg {
            #expect(data["lod"] as? Int == 2)
            #expect(data["prev"] as? Int == 1)
        } else {
            Issue.record("Expected debugLod message")
        }
    }

    @Test("debug_activity parses activity observability payload")
    func debugActivityParsing() {
        let json = "{\"type\":\"debug_activity\",\"data\":{\"event\":\"activity_start\",\"state\":\"user_speaking\",\"queue_status\":\"forwarded\",\"event_count\":1}}"
        let msg = DownstreamMessage.parse(text: json)

        if case .debugActivity(let data) = msg {
            #expect(data["event"] as? String == "activity_start")
            #expect(data["state"] as? String == "user_speaking")
            #expect(data["queue_status"] as? String == "forwarded")
            #expect(data["event_count"] as? Int == 1)
        } else {
            Issue.record("Expected debugActivity message")
        }
    }

    @Test("interrupted message parses")
    func interruptedParsing() {
        let json = "{\"type\":\"interrupted\",\"message\":\"Model output was interrupted.\"}"
        let msg = DownstreamMessage.parse(text: json)

        if case .interrupted = msg {
            // pass
        } else {
            Issue.record("Expected interrupted message")
        }
    }
}

@Suite("ToolBehaviorMode")
struct ToolBehaviorModeTests {

    @Test("parses INTERRUPT from top-level behavior")
    func parseInterrupt() {
        let json: [String: Any] = ["behavior": "INTERRUPT"]
        #expect(ToolBehaviorMode.parse(from: json) == .INTERRUPT)
    }

    @Test("parses WHEN_IDLE from nested data.behavior")
    func parseNestedWhenIdle() {
        let json: [String: Any] = ["data": ["behavior": "WHEN_IDLE"]]
        #expect(ToolBehaviorMode.parse(from: json) == .WHEN_IDLE)
    }

    @Test("parses SILENT case-insensitively")
    func parseSilentCaseInsensitive() {
        let json: [String: Any] = ["behavior": "silent"]
        #expect(ToolBehaviorMode.parse(from: json) == .SILENT)
    }

    @Test("defaults to WHEN_IDLE when behavior missing")
    func defaultsToWhenIdle() {
        let json: [String: Any] = ["type": "test"]
        #expect(ToolBehaviorMode.parse(from: json) == .WHEN_IDLE)
    }
}

@Suite("TelemetryData Codable")
struct TelemetryDataTests {

    @Test("round-trip encode/decode")
    func roundTrip() throws {
        var telemetry = TelemetryData()
        telemetry.motionState = "running"
        telemetry.stepCadence = 160
        telemetry.ambientNoiseDb = 72.0
        telemetry.gps = GPSData(lat: 37.7749, lng: -122.4194, accuracy: 5.0, speed: 2.5, altitude: 10.0)
        telemetry.heading = 180.0
        telemetry.heartRate = 120.0

        let encoder = JSONEncoder()
        let data = try encoder.encode(telemetry)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(TelemetryData.self, from: data)

        #expect(decoded.motionState == "running")
        #expect(decoded.stepCadence == 160)
        #expect(decoded.ambientNoiseDb == 72.0)
        #expect(decoded.gps?.lat == 37.7749)
        #expect(decoded.gps?.lng == -122.4194)
        #expect(decoded.heading == 180.0)
        #expect(decoded.heartRate == 120.0)
    }

    @Test("snake_case coding keys")
    func codingKeys() throws {
        var telemetry = TelemetryData()
        telemetry.motionState = "walking"
        telemetry.heartRate = 80.0

        let data = try JSONEncoder().encode(telemetry)
        let jsonStr = String(data: data, encoding: .utf8)!

        #expect(jsonStr.contains("motion_state"))
        #expect(jsonStr.contains("heart_rate"))
        #expect(jsonStr.contains("step_cadence"))
        #expect(jsonStr.contains("ambient_noise_db"))
        #expect(!jsonStr.contains("motionState"))
    }

    @Test("default values are sensible")
    func defaultValues() {
        let telemetry = TelemetryData()
        #expect(telemetry.motionState == "stationary")
        #expect(telemetry.stepCadence == 0)
        #expect(telemetry.ambientNoiseDb == 50.0)
        #expect(telemetry.gps == nil)
        #expect(telemetry.heartRate == nil)
    }
}

@Suite("GPSData Codable")
struct GPSDataTests {

    @Test("uses latitude/longitude as coding keys")
    func codingKeys() throws {
        let gps = GPSData(lat: 40.7128, lng: -74.0060, accuracy: 10.0, speed: nil, altitude: nil)
        let data = try JSONEncoder().encode(gps)
        let jsonStr = String(data: data, encoding: .utf8)!

        #expect(jsonStr.contains("latitude"))
        #expect(jsonStr.contains("longitude"))
        #expect(!jsonStr.contains("\"lat\""))
        #expect(!jsonStr.contains("\"lng\""))
    }
}
