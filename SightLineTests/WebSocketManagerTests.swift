//
//  WebSocketManagerTests.swift
//  SightLineTests
//
//  Smoke tests for WebSocket connection lifecycle, reconnection,
//  and message handling.
//

import Testing
import Foundation
@testable import SightLine

@Suite("WebSocket Manager Tests")
struct WebSocketManagerTests {

    @Test("WebSocketManager initializes with disconnected state")
    func initialState() {
        let manager = WebSocketManager()
        // WebSocketManager should start disconnected
        #expect(manager.isConnected == false)
    }

    @Test("WebSocketManager exposes callback properties")
    func callbackProperties() {
        let manager = WebSocketManager()
        // Verify callback properties exist and are settable
        manager.onAudioReceived = { _ in }
        manager.onTextReceived = { _ in }
        manager.onTextSent = { _ in }
        manager.onConnectionStateChanged = { _ in }
        manager.onDisconnectionDegraded = { }
        manager.onConnectionRestored = { }
        // No crash = pass
    }

    @Test("WebSocketManager handles disconnect gracefully")
    func disconnectGracefully() {
        let manager = WebSocketManager()
        // Calling disconnect on an unconnected manager should not crash
        manager.disconnect()
        #expect(manager.isConnected == false)
    }

    @Test("WebSocketManager sendText does not crash when disconnected")
    func sendTextWhenDisconnected() {
        let manager = WebSocketManager()
        // Sending text when not connected should be a no-op, not a crash
        manager.sendText("{\"type\":\"audio\",\"data\":\"\"}")
    }
}
