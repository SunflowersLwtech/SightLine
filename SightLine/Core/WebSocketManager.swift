//
//  WebSocketManager.swift
//  SightLine
//
//  WebSocket client using Apple's Network framework (NWConnection).
//  Handles connection lifecycle, automatic reconnection with exponential backoff,
//  and network path monitoring for seamless WiFi/Cellular transitions.
//

import Network
import Foundation
import Combine
import UIKit
import os

class WebSocketManager: ObservableObject {
    @Published var isConnected = false

    private static let logger = Logger(subsystem: "com.sightline.app", category: "WebSocket")

    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "com.sightline.websocket", qos: .userInitiated)
    private var reconnectDelay: TimeInterval = 1.0
    private let maxReconnectDelay: TimeInterval = 30.0
    private var serverURL: URL?
    private var intentionalDisconnect = false
    private var pathMonitor: NWPathMonitor?
    private var reconnectWorkItem: DispatchWorkItem?
    private var pingTimer: DispatchSourceTimer?
    private let pingInterval: TimeInterval = 15.0
    private var isConnectionReady = false
    private var isConnectionInProgress = false

    /// Tracks when connection entered .ready state — used to detect short-lived connections
    /// where the server crashes immediately after accepting the WebSocket handshake.
    private var lastConnectionReadyTime: DispatchTime?
    /// Minimum connection duration (seconds) before we consider it "stable" and reset backoff.
    private let minStableConnectionDuration: TimeInterval = 5.0

    // Callbacks
    var onAudioReceived: ((Data) -> Void)?
    var onTextReceived: ((String) -> Void)?
    var onTextSent: ((String) -> Void)?
    var onConnectionStateChanged: ((Bool) -> Void)?
    /// Called when connection is lost — UI should enter safe mode (LOD 1).
    var onDisconnectionDegraded: (() -> Void)?
    /// Called when connection is restored after a disconnection.
    var onConnectionRestored: (() -> Void)?
    /// Called when server reports a fatal error (close code 1008) — do NOT reconnect.
    var onFatalServerError: ((String) -> Void)?

    /// Track whether we have notified about degradation to avoid duplicates.
    private var hasDegradedNotification = false
    /// Set when server sends close code 1008 (fatal error) — suppresses auto-reconnect.
    private var receivedFatalClose = false

    func connect(url: URL) {
        serverURL = url
        intentionalDisconnect = false
        receivedFatalClose = false
        reconnectDelay = 1.0
        lastConnectionReadyTime = nil
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        stopPingTimer()
        connection?.cancel()
        connection = nil
        isConnectionReady = false
        isConnectionInProgress = false
        startConnection(url: url)
        // PathMonitor is started in .ready handler to avoid race condition:
        // NWPathMonitor fires immediately on start(), which can cancel
        // a connection still in TLS handshake (.preparing state).
    }

    func disconnect() {
        intentionalDisconnect = true
        lastConnectionReadyTime = nil
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        pathMonitor?.cancel()
        pathMonitor = nil
        stopPingTimer()
        connection?.cancel()
        connection = nil
        isConnectionReady = false
        isConnectionInProgress = false
        updateConnectionState(false)
    }

    func sendText(_ text: String) {
        guard isConnectionReady, let activeConnection = connection else { return }
        guard let data = text.data(using: .utf8) else { return }
        DispatchQueue.main.async { [weak self] in
            self?.onTextSent?(text)
        }
        let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
        let context = NWConnection.ContentContext(identifier: "textMessage",
                                                  metadata: [metadata])
        activeConnection.send(content: data, contentContext: context, isComplete: true,
                        completion: .contentProcessed { [weak self] error in
            if let error = error {
                Self.logger.error("WebSocket send text error: \(error)")
                self?.handleDisconnect(sourceConnection: activeConnection)
            }
        })
    }

    func sendBinary(_ data: Data) {
        guard isConnectionReady, let activeConnection = connection else { return }
        let metadata = NWProtocolWebSocket.Metadata(opcode: .binary)
        let context = NWConnection.ContentContext(identifier: "binaryMessage",
                                                  metadata: [metadata])
        activeConnection.send(content: data, contentContext: context, isComplete: true,
                        completion: .contentProcessed { [weak self] error in
            if let error = error {
                Self.logger.error("WebSocket send binary error: \(error)")
                self?.handleDisconnect(sourceConnection: activeConnection)
            }
        })
    }

    /// Explicit reconnect path used for server-initiated GoAway messages.
    func reconnect(afterMs: Int = 0) {
        guard let url = serverURL else { return }

        intentionalDisconnect = false
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        stopPingTimer()
        connection?.cancel()
        connection = nil
        isConnectionReady = false
        isConnectionInProgress = false
        updateConnectionState(false)

        let delay = max(0.0, Double(afterMs) / 1000.0)
        let workItem = DispatchWorkItem { [weak self] in
            guard let self = self, !self.intentionalDisconnect else { return }
            self.reconnectWorkItem = nil
            self.startConnection(url: url)
        }
        reconnectWorkItem = workItem
        queue.asyncAfter(deadline: .now() + delay, execute: workItem)
    }

    // MARK: - Private

    private func startConnection(url: URL) {
        guard !intentionalDisconnect else { return }

        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        stopPingTimer()

        let useTLS = url.scheme == "wss"
        let parameters = useTLS ? NWParameters.tls : NWParameters.tcp
        let wsOptions = NWProtocolWebSocket.Options()
        wsOptions.autoReplyPing = true
        parameters.defaultProtocolStack.applicationProtocols.insert(wsOptions, at: 0)

        let newConnection = NWConnection(to: .url(url), using: parameters)
        self.connection = newConnection
        isConnectionInProgress = true
        isConnectionReady = false

        newConnection.stateUpdateHandler = { [weak self, weak newConnection] state in
            guard let self = self else { return }
            guard let activeConnection = newConnection else { return }
            guard self.connection === activeConnection else { return }

            switch state {
            case .preparing:
                self.isConnectionInProgress = true
                self.isConnectionReady = false

            case .ready:
                Self.logger.info("WebSocket connected")
                self.isConnectionInProgress = false
                self.isConnectionReady = true
                self.lastConnectionReadyTime = .now()
                // Don't reset reconnectDelay here — defer to handleDisconnect
                // which checks connection duration to avoid rapid-reconnect loops
                // when the server crashes immediately after accepting.
                self.updateConnectionState(true)
                self.startPathMonitor()
                self.startPingTimer(for: activeConnection)
                self.receiveLoop(connection: activeConnection)

                // Notify restoration if we were degraded
                if self.hasDegradedNotification {
                    self.hasDegradedNotification = false
                    DispatchQueue.main.async {
                        HapticManager.shared.connectionRestored()
                        self.onConnectionRestored?()
                    }
                }

            case .failed(let error):
                Self.logger.error("WebSocket failed: \(error)")
                self.isConnectionInProgress = false
                self.isConnectionReady = false
                self.updateConnectionState(false)
                self.stopPingTimer()
                self.handleDisconnect(sourceConnection: activeConnection)

            case .cancelled:
                Self.logger.info("WebSocket cancelled")
                self.isConnectionInProgress = false
                self.isConnectionReady = false
                self.stopPingTimer()
                self.updateConnectionState(false)

            case .waiting(let error):
                Self.logger.warning("WebSocket waiting: \(error)")
                self.isConnectionInProgress = true
                self.isConnectionReady = false
                self.updateConnectionState(false)

            default:
                break
            }
        }

        newConnection.start(queue: queue)
    }

    private func receiveLoop(connection: NWConnection) {
        connection.receiveMessage { [weak self, weak connection] content, context, isComplete, error in
            guard let self = self else { return }
            guard let activeConnection = connection else { return }
            guard self.connection === activeConnection else { return }

            if let error = error {
                Self.logger.error("WebSocket receive error: \(error)")
                self.handleDisconnect(sourceConnection: activeConnection)
                return
            }

            // Determine message type from WebSocket metadata
            if let metadata = context?.protocolMetadata(definition: NWProtocolWebSocket.definition)
                as? NWProtocolWebSocket.Metadata {
                switch metadata.opcode {
                case .binary:
                    if let data = content {
                        self.onAudioReceived?(data)
                    }
                case .text:
                    if let data = content, let text = String(data: data, encoding: .utf8) {
                        self.onTextReceived?(text)
                    }
                case .close:
                    // Close code 1008 = fatal server error — do not auto-reconnect
                    if metadata.closeCode == .protocolCode(.policyViolation) {
                        Self.logger.error("Server sent fatal close (1008) — suppressing reconnect")
                        self.receivedFatalClose = true
                        self.isConnectionReady = false
                        self.isConnectionInProgress = false
                        self.stopPingTimer()
                        self.updateConnectionState(false)
                        DispatchQueue.main.async {
                            self.onFatalServerError?("Fatal server error. Please restart the app.")
                        }
                        return
                    }
                    self.handleDisconnect(sourceConnection: activeConnection)
                    return
                default:
                    break
                }
            }

            if isComplete, context == nil, (content == nil || content?.isEmpty == true) {
                self.handleDisconnect(sourceConnection: activeConnection)
                return
            }

            // Continue receiving
            self.receiveLoop(connection: activeConnection)
        }
    }

    private func handleDisconnect(sourceConnection: NWConnection? = nil) {
        guard !intentionalDisconnect, !receivedFatalClose else { return }
        if let sourceConnection, let currentConnection = connection, sourceConnection !== currentConnection {
            return
        }
        guard reconnectWorkItem == nil else { return }

        isConnectionInProgress = false
        isConnectionReady = false
        updateConnectionState(false)
        stopPingTimer()

        // Haptic feedback + degradation notification
        if !hasDegradedNotification {
            hasDegradedNotification = true
            DispatchQueue.main.async { [weak self] in
                HapticManager.shared.connectionLost()
                self?.onDisconnectionDegraded?()
            }
        }

        // Determine if this was a short-lived connection (server crash after accept).
        // Only reset backoff when the connection was genuinely stable (> minStableConnectionDuration).
        if let readyTime = lastConnectionReadyTime {
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - readyTime.uptimeNanoseconds) / 1_000_000_000
            if elapsed >= minStableConnectionDuration {
                // Stable connection — reset backoff for next reconnect
                reconnectDelay = 1.0
            }
            // else: short-lived — keep current backoff and let it grow below
        }
        // If lastConnectionReadyTime is nil, connection never reached .ready — keep backoff as-is

        // Exponential backoff reconnection
        let delay = reconnectDelay
        reconnectDelay = min(reconnectDelay * 2, maxReconnectDelay)

        Self.logger.info("Reconnecting in \(delay)s...")
        let workItem = DispatchWorkItem { [weak self] in
            guard let self = self, !self.intentionalDisconnect,
                  let url = self.serverURL else { return }
            self.reconnectWorkItem = nil
            self.connection?.cancel()
            self.startConnection(url: url)
        }
        reconnectWorkItem = workItem
        queue.asyncAfter(deadline: .now() + delay, execute: workItem)
    }

    private func updateConnectionState(_ connected: Bool) {
        DispatchQueue.main.async { [weak self] in
            self?.isConnected = connected
            self?.onConnectionStateChanged?(connected)
        }
    }

    /// Monitor network path changes (WiFi <-> Cellular) for seamless transitions
    private func startPathMonitor() {
        pathMonitor?.cancel()
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self = self, !self.intentionalDisconnect else { return }

            if path.status == .satisfied {
                // Network available - reconnect if not already connected
                if !self.isConnectionReady && !self.isConnectionInProgress,
                   self.reconnectWorkItem == nil,
                   let url = self.serverURL {
                    Self.logger.info("Network path changed, reconnecting...")
                    self.reconnectDelay = 1.0
                    self.connection?.cancel()
                    self.startConnection(url: url)
                }
            } else {
                Self.logger.warning("Network path unsatisfied")
                self.isConnectionReady = false
                self.isConnectionInProgress = false
                self.stopPingTimer()
                self.updateConnectionState(false)
            }
        }
        monitor.start(queue: queue)
        pathMonitor = monitor
    }

    private func startPingTimer(for connection: NWConnection) {
        stopPingTimer()

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + pingInterval, repeating: pingInterval)
        timer.setEventHandler { [weak self, weak connection] in
            guard let self = self else { return }
            guard let activeConnection = connection else { return }
            guard self.connection === activeConnection, self.isConnectionReady else { return }

            let metadata = NWProtocolWebSocket.Metadata(opcode: .ping)
            let context = NWConnection.ContentContext(identifier: "ping", metadata: [metadata])
            activeConnection.send(content: nil, contentContext: context, isComplete: true,
                                  completion: .contentProcessed { [weak self] error in
                if let error = error {
                    Self.logger.error("WebSocket ping error: \(error)")
                    self?.handleDisconnect(sourceConnection: activeConnection)
                }
            })
        }
        pingTimer = timer
        timer.resume()
    }

    private func stopPingTimer() {
        pingTimer?.cancel()
        pingTimer = nil
    }
}
