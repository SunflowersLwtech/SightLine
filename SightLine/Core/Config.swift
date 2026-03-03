//
//  Config.swift
//  SightLine
//
//  Central configuration for server URLs, audio/video parameters, and defaults.
//

import Foundation

enum SightLineConfig {
    // Server URL — Debug builds connect to local Mac, Release to Cloud Run
    #if DEBUG
    static let serverBaseURL = "ws://Lius-MacBook-Air.local:8100"
    #else
    static let serverBaseURL = "wss://sightline-backend-kp47ssyf4q-uc.a.run.app"
    #endif
    static let sessionResumptionHandleDefaultsKey = "sightline.session_resumption_handle"

    // WebSocket path template
    static func wsURL(userId: String, sessionId: String, resumeHandle: String? = nil) -> URL? {
        guard var components = URLComponents(string: "\(serverBaseURL)/ws/\(userId)/\(sessionId)") else {
            return nil
        }
        if let resumeHandle, !resumeHandle.isEmpty {
            components.queryItems = [URLQueryItem(name: "resume_handle", value: resumeHandle)]
        }
        return components.url
    }

    // Audio
    static let audioInputSampleRate: Double = 16000   // Gemini requires 16kHz
    static let audioOutputSampleRate: Double = 24000   // Gemini outputs 24kHz
    static let audioBufferSize: UInt32 = 1600          // ~33ms at 48kHz hw input; also converter output capacity
    static let audioJitterBufferChunks: Int = 2        // reduced startup latency (was 3→2, saves ~100ms)
    static let audioScheduleAheadCount: Int = 3        // reduced sliding window (was 4)
    static let audioJitterMaxWait: TimeInterval = 0.08 // 80ms fallback — tolerate more network jitter
    static let audioMaxPendingChunks: Int = 40         // overflow guard (~4s at 100ms/chunk)

    // Video
    static let videoFrameWidth: Int = 768
    static let videoFrameHeight: Int = 768
    static let jpegQuality: CGFloat = 0.7
    static let defaultFrameInterval: TimeInterval = 1.0  // 1 FPS

    // User defaults (mutable for demo user switching)
    static var defaultUserId: String {
        get { UserDefaults.standard.string(forKey: "sightline.current_user_id") ?? "default_user" }
        set { UserDefaults.standard.set(newValue, forKey: "sightline.current_user_id") }
    }
    private static var _sessionId: String = UUID().uuidString.lowercased()
    static var defaultSessionId: String {
        get { _sessionId }
        set { _sessionId = newValue }
    }
}
