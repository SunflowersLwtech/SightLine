//
//  MessageProtocol.swift
//  SightLine
//
//  Defines the WebSocket message protocol for communication with the backend.
//  Upstream = iOS -> Server, Downstream = Server -> iOS.
//

import Foundation

// MARK: - Upstream Messages (iOS -> Server)

/// Magic bytes for binary WebSocket protocol.
/// Eliminates ~33% Base64 overhead on audio/image payloads.
enum BinaryMagic {
    static let audio: UInt8 = 0x01   // PCM 16kHz mono
    static let image: UInt8 = 0x02   // JPEG 768x768
}

enum UpstreamMessage {
    case audio(data: Data)                         // PCM 16kHz mono
    case image(data: Data, mimeType: String)       // JPEG 768x768
    case telemetry(data: TelemetryData)
    case gesture(type: String)
    case reloadFaceLibrary
    case clearFaceLibrary
    case cameraFailure(error: String)
    case muteToggle(muted: Bool)
    case pause(paused: Bool)
    case cameraToggle(active: Bool)
    case playbackDrained
    case clientBargeIn

    /// Encode as optimized binary frame (magic byte + raw payload).
    /// Returns nil for message types that must be sent as JSON text.
    func toBinary() -> Data? {
        switch self {
        case .audio(let data):
            var frame = Data(capacity: 1 + data.count)
            frame.append(BinaryMagic.audio)
            frame.append(data)
            return frame
        case .image(let data, _):
            var frame = Data(capacity: 1 + data.count)
            frame.append(BinaryMagic.image)
            frame.append(data)
            return frame
        default:
            return nil  // Non-binary types use JSON
        }
    }

    /// Legacy JSON encoding (for telemetry, gestures, control messages).
    func toJSON() -> String {
        switch self {
        case .audio(let data):
            return "{\"type\":\"audio\",\"data\":\"\(data.base64EncodedString())\"}"
        case .image(let data, let mimeType):
            return "{\"type\":\"image\",\"data\":\"\(data.base64EncodedString())\",\"mimeType\":\"\(mimeType)\"}"
        case .telemetry(let data):
            guard let jsonData = try? JSONEncoder().encode(data),
                  let jsonStr = String(data: jsonData, encoding: .utf8) else {
                return "{\"type\":\"telemetry\",\"data\":{}}"
            }
            return "{\"type\":\"telemetry\",\"data\":\(jsonStr)}"
        case .gesture(let type):
            return "{\"type\":\"gesture\",\"gesture\":\"\(type)\"}"
        case .reloadFaceLibrary:
            return "{\"type\":\"reload_face_library\"}"
        case .clearFaceLibrary:
            return "{\"type\":\"clear_face_library\"}"
        case .cameraFailure(let error):
            let escaped = error.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return "{\"type\":\"camera_failure\",\"error\":\"\(escaped)\",\"reason\":\"\(escaped)\"}"
        case .muteToggle(let muted):
            return "{\"type\":\"gesture\",\"gesture\":\"mute_toggle\",\"muted\":\(muted)}"
        case .pause(let paused):
            return "{\"type\":\"gesture\",\"gesture\":\"pause\",\"paused\":\(paused)}"
        case .cameraToggle(let active):
            return "{\"type\":\"gesture\",\"gesture\":\"camera_toggle\",\"active\":\(active)}"
        case .playbackDrained:
            return "{\"type\":\"playback_drained\"}"
        case .clientBargeIn:
            return "{\"type\":\"client_barge_in\"}"
        }
    }
}

// MARK: - Downstream Messages (Server -> iOS)

enum ToolBehaviorMode: String {
    case INTERRUPT
    case WHEN_IDLE
    case SILENT

    static func parse(from json: [String: Any]) -> ToolBehaviorMode {
        if let behavior = json["behavior"] as? String,
           let mode = ToolBehaviorMode(rawValue: behavior.uppercased()) {
            return mode
        }
        if let data = json["data"] as? [String: Any],
           let behavior = data["behavior"] as? String,
           let mode = ToolBehaviorMode(rawValue: behavior.uppercased()) {
            return mode
        }
        return .WHEN_IDLE
    }
}

enum DownstreamMessage {
    case audio(data: Data)                          // PCM 24kHz
    case transcript(text: String, role: String)     // "user" or "agent"
    case lodUpdate(lod: Int)
    case goAway(retryMs: Int)
    case sessionResumption(handle: String)
    case sessionReady                               // Gemini Live API ready
    case faceLibraryReloaded(count: Int)
    case faceLibraryCleared(deletedCount: Int)
    case error(message: String)
    case toolEvent(tool: String, behavior: ToolBehaviorMode, payload: [String: Any])
    case visionResult(summary: String, behavior: ToolBehaviorMode)
    case ocrResult(summary: String, behavior: ToolBehaviorMode)
    case visionDebug(data: [String: Any])
    case ocrDebug(data: [String: Any])
    case faceDebug(data: [String: Any])
    case frameAck(frameId: Int, queuedAgents: [String])
    case navigationResult(summary: String, behavior: ToolBehaviorMode)
    case searchResult(summary: String, behavior: ToolBehaviorMode)
    case personIdentified(name: String, behavior: ToolBehaviorMode)
    case identityUpdate(name: String, matched: Bool, behavior: ToolBehaviorMode)
    case capabilityDegraded(capability: String, reason: String, recoverable: Bool)
    case debugLod(data: [String: Any])
    case debugActivity(data: [String: Any])
    case interrupted
    case profileUpdatedAck
    case toolsManifest(tools: [[String: Any]], contextModules: [[String: Any]], subAgents: [String: String])
    case unknown(raw: String)

    static func parse(text: String) -> DownstreamMessage? {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            return .unknown(raw: text)
        }

        let behavior = ToolBehaviorMode.parse(from: json)
        let dataPayload = json["data"] as? [String: Any] ?? [:]

        func extractSummary() -> String {
            if let summary = json["summary"] as? String, !summary.isEmpty {
                return summary
            }
            if let summary = dataPayload["summary"] as? String, !summary.isEmpty {
                return summary
            }
            if let text = dataPayload["text"] as? String, !text.isEmpty {
                return text
            }
            return ""
        }

        func extractPersonName() -> String {
            if let name = json["person_name"] as? String, !name.isEmpty {
                return name
            }
            if let name = dataPayload["person_name"] as? String, !name.isEmpty {
                return name
            }
            if let name = json["name"] as? String, !name.isEmpty {
                return name
            }
            return "unknown"
        }

        switch type {
        case "transcript":
            let text = json["text"] as? String ?? ""
            let role = json["role"] as? String ?? "agent"
            return .transcript(text: text, role: role)
        case "lod_update":
            let lod = json["lod"] as? Int ?? 2
            return .lodUpdate(lod: lod)
        case "go_away":
            let retryMs = json["retry_ms"] as? Int ?? 500
            return .goAway(retryMs: retryMs)
        case "session_resumption":
            let handle = json["handle"] as? String ?? ""
            return .sessionResumption(handle: handle)
        case "session_ready":
            return .sessionReady
        case "face_library_reloaded":
            let count = (json["count"] as? Int) ?? (dataPayload["count"] as? Int) ?? 0
            return .faceLibraryReloaded(count: count)
        case "face_library_cleared":
            let deletedCount = (json["deleted_count"] as? Int) ?? (dataPayload["deleted_count"] as? Int) ?? 0
            return .faceLibraryCleared(deletedCount: deletedCount)
        case "error":
            let message = (json["error"] as? String)
                ?? (json["message"] as? String)
                ?? (dataPayload["error"] as? String)
                ?? "Unknown server error."
            return .error(message: message)
        case "tool_event", "tool_result", "tool_status":
            let tool = (json["tool"] as? String)
                ?? (json["name"] as? String)
                ?? (dataPayload["tool"] as? String)
                ?? "unknown_tool"
            return .toolEvent(tool: tool, behavior: behavior, payload: dataPayload)
        case "vision_result":
            return .visionResult(summary: extractSummary(), behavior: behavior)
        case "ocr_result":
            return .ocrResult(summary: extractSummary(), behavior: behavior)
        case "vision_debug":
            return .visionDebug(data: json["data"] as? [String: Any] ?? json)
        case "ocr_debug":
            return .ocrDebug(data: json["data"] as? [String: Any] ?? json)
        case "face_debug":
            return .faceDebug(data: json["data"] as? [String: Any] ?? json)
        case "frame_ack":
            let frameId = (json["frame_id"] as? Int) ?? (dataPayload["frame_id"] as? Int) ?? -1
            let queuedAgents = (json["queued_agents"] as? [String]) ?? (dataPayload["queued_agents"] as? [String]) ?? []
            return .frameAck(frameId: frameId, queuedAgents: queuedAgents)
        case "navigation_result", "navigate_result":
            return .navigationResult(summary: extractSummary(), behavior: behavior)
        case "search_result", "grounding_result":
            return .searchResult(summary: extractSummary(), behavior: behavior)
        case "person_identified":
            return .personIdentified(name: extractPersonName(), behavior: behavior)
        case "identity_update":
            let matched = (json["matched"] as? Bool)
                ?? (dataPayload["matched"] as? Bool)
                ?? false
            return .identityUpdate(name: extractPersonName(), matched: matched, behavior: behavior)
        case "capability_degraded":
            let capability = json["capability"] as? String ?? "unknown"
            let reason = json["reason"] as? String ?? ""
            let recoverable = json["recoverable"] as? Bool ?? true
            return .capabilityDegraded(capability: capability, reason: reason, recoverable: recoverable)
        case "debug_lod":
            let lodData = json["data"] as? [String: Any] ?? json
            return .debugLod(data: lodData)
        case "debug_activity":
            let activityData = json["data"] as? [String: Any] ?? json
            return .debugActivity(data: activityData)
        case "interrupted":
            return .interrupted
        case "profile_updated_ack":
            return .profileUpdatedAck
        case "tools_manifest":
            let tools = json["tools"] as? [[String: Any]] ?? []
            let modules = json["context_modules"] as? [[String: Any]] ?? []
            let agents = json["sub_agents"] as? [String: String] ?? [:]
            return .toolsManifest(tools: tools, contextModules: modules, subAgents: agents)
        default:
            return .unknown(raw: text)
        }
    }
}

// MARK: - Telemetry Data

struct TelemetryData: Codable {
    var motionState: String = "stationary"
    var stepCadence: Int = 0
    var ambientNoiseDb: Double = 50.0
    var gps: GPSData?
    var heading: Double?
    var timeContext: String = "unknown"
    var heartRate: Double?
    var userGesture: String?
    var deviceType: String = "phone_only"
    var weather: WeatherManager.WeatherSnapshot?
    var depth: DepthEstimator.DepthSummary?

    // Watch extended context
    var watchPitch: Double?
    var watchRoll: Double?
    var watchYaw: Double?
    var watchStabilityScore: Double?
    var watchHeading: Double?
    var watchHeadingAccuracy: Double?
    var spO2: Double?
    var watchNoiseExposure: Double?

    enum CodingKeys: String, CodingKey {
        case motionState = "motion_state"
        case stepCadence = "step_cadence"
        case ambientNoiseDb = "ambient_noise_db"
        case gps
        case heading
        case timeContext = "time_context"
        case heartRate = "heart_rate"
        case userGesture = "user_gesture"
        case deviceType = "device_type"
        case weather
        case depth
        case watchPitch = "watch_pitch"
        case watchRoll = "watch_roll"
        case watchYaw = "watch_yaw"
        case watchStabilityScore = "watch_stability_score"
        case watchHeading = "watch_heading"
        case watchHeadingAccuracy = "watch_heading_accuracy"
        case spO2 = "sp_o2"
        case watchNoiseExposure = "watch_noise_exposure"
    }
}

struct GPSData: Codable {
    var lat: Double
    var lng: Double
    var accuracy: Double?
    var speed: Double?
    var altitude: Double?

    enum CodingKeys: String, CodingKey {
        case lat = "latitude"
        case lng = "longitude"
        case accuracy
        case speed
        case altitude
    }
}
