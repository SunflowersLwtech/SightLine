//
//  ConfigTests.swift
//  SightLineTests
//
//  Tests for SightLineConfig constants and URL generation.
//

import Testing
import Foundation
@testable import SightLine

@Suite("SightLineConfig")
struct ConfigTests {

    @Test("wsURL generates correct WebSocket URL")
    func wsUrlGeneration() {
        let url = SightLineConfig.wsURL(userId: "user123", sessionId: "sess456")
        #expect(url != nil)
        #expect(url!.absoluteString.contains("/ws/user123/sess456"))
        #if DEBUG
        #expect(url!.scheme == "ws")
        #else
        #expect(url!.scheme == "wss")
        #endif
    }

    @Test("wsURL includes resume handle query item")
    func wsURLIncludesResumeHandle() {
        let url = SightLineConfig.wsURL(
            userId: "user123",
            sessionId: "sess456",
            resumeHandle: "resume-token-123"
        )
        #expect(url != nil)
        let components = URLComponents(url: url!, resolvingAgainstBaseURL: false)
        let resumeHandle = components?.queryItems?.first(where: { $0.name == "resume_handle" })?.value
        #expect(resumeHandle == "resume-token-123")
    }

    @Test("server base URL uses correct scheme for build configuration")
    func serverBaseUrlScheme() {
        #if DEBUG
        #expect(SightLineConfig.serverBaseURL.hasPrefix("ws://"))
        #else
        #expect(SightLineConfig.serverBaseURL.hasPrefix("wss://"))
        #endif
    }

    @Test("audio input sample rate is 16kHz for Gemini")
    func audioInputSampleRate() {
        #expect(SightLineConfig.audioInputSampleRate == 16000)
    }

    @Test("audio output sample rate is 24kHz from Gemini")
    func audioOutputSampleRate() {
        #expect(SightLineConfig.audioOutputSampleRate == 24000)
    }

    @Test("audio buffer size gives ~33ms at 48kHz hardware input")
    func audioBufferSize() {
        // 48000 samples/sec * 0.033 sec ≈ 1600 samples (Google recommends 20-40ms)
        #expect(SightLineConfig.audioBufferSize == 1600)
    }

    @Test("audio jitter buffer keeps 2-4 chunks")
    func audioJitterBufferChunks() {
        #expect((2...4).contains(SightLineConfig.audioJitterBufferChunks))
    }

    @Test("audio schedule-ahead count is 2-4 buffers")
    func audioScheduleAheadCount() {
        #expect((2...4).contains(SightLineConfig.audioScheduleAheadCount))
    }

    @Test("audio jitter max wait is under 100ms")
    func audioJitterMaxWait() {
        #expect(SightLineConfig.audioJitterMaxWait > 0)
        #expect(SightLineConfig.audioJitterMaxWait <= 0.1)
    }

    @Test("video frame dimensions are 768x768")
    func videoFrameDimensions() {
        #expect(SightLineConfig.videoFrameWidth == 768)
        #expect(SightLineConfig.videoFrameHeight == 768)
    }

    @Test("JPEG quality is 0.7")
    func jpegQuality() {
        #expect(SightLineConfig.jpegQuality == 0.7)
    }

    @Test("default frame interval is 1.0s")
    func defaultFrameInterval() {
        #expect(SightLineConfig.defaultFrameInterval == 1.0)
    }

    @Test("default user ID is set")
    func defaultUserId() {
        #expect(!SightLineConfig.defaultUserId.isEmpty)
    }

    @Test("default session ID is a UUID")
    func defaultSessionId() {
        #expect(!SightLineConfig.defaultSessionId.isEmpty)
        // UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        #expect(SightLineConfig.defaultSessionId.count == 36)
        #expect(SightLineConfig.defaultSessionId == SightLineConfig.defaultSessionId.lowercased())
    }
}
