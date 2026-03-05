//
//  Config.swift
//  SightLine
//
//  Central configuration for server URLs, audio/video parameters, and defaults.
//

import Foundation

enum SightLineConfig {
    private static let debugServerBaseURLDefaultsKey = "sightline.debug_server_base_url"

    // Server URL — Debug builds support runtime override for real-device local testing.
    #if DEBUG
    static let serverBaseURL: String = {
        // Highest priority: launch-time environment variable from `devicectl`.
        if let runtime = ProcessInfo.processInfo.environment["SIGHTLINE_WS_BASE_URL"],
           !runtime.isEmpty {
            // Persist once so manual relaunches keep using the same LAN endpoint.
            UserDefaults.standard.set(runtime, forKey: debugServerBaseURLDefaultsKey)
            return runtime
        }
        // Next priority: previously persisted LAN endpoint.
        if let persisted = UserDefaults.standard.string(forKey: debugServerBaseURLDefaultsKey),
           !persisted.isEmpty {
            return persisted
        }
        // Allows CLI deploy scripts to inject current LAN IP at build time:
        // `INFOPLIST_KEY_SIGHTLINE_WS_BASE_URL=ws://<ip>:8100`.
        if let injected = Bundle.main.object(forInfoDictionaryKey: "SIGHTLINE_WS_BASE_URL") as? String,
           !injected.isEmpty {
            return injected
        }
        return "ws://Lius-MacBook-Air.local:8100"
    }()
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
