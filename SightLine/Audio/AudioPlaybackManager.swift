//
//  AudioPlaybackManager.swift
//  SightLine
//
//  Plays PCM 24kHz mono 16-bit audio received from the Gemini backend.
//  Supports immediate stop for barge-in (user interrupting the agent).
//
//  Uses SharedAudioEngine.shared.playerNode so that playback and capture
//  share a single audio graph, enabling hardware AEC.
//

import AVFoundation
import Combine
import os

enum AudioPriority: Int, Comparable {
    case low = 0        // ambient context, can be dropped
    case normal = 1     // standard speech (Gemini stream)
    case high = 2       // navigation instructions during active routing
    case critical = 3   // safety hazards — interrupts everything

    static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

class AudioPlaybackManager: ObservableObject {
    @Published var isPlaying = false

    /// Called when audio buffer overflow occurs; parameter is the number of dropped chunks.
    var onBufferOverflow: ((Int) -> Void)?

    /// Called when all buffered audio has finished playing (drain completed).
    /// Used to signal the server that it is safe to flush new context injections.
    var onPlaybackDrained: (() -> Void)?

    private static let logger = Logger(subsystem: "com.sightline.app", category: "AudioPlayback")

    private var playbackFormat: AVAudioFormat?
    private let schedulingQueue = DispatchQueue(
        label: "com.sightline.audio.playback.scheduling",
        qos: .userInitiated
    )
    private var criticalChunks: [Data] = []
    private var highChunks: [Data] = []
    private var normalChunks: [Data] = []
    private var lowChunks: [Data] = []
    private var isDrainActive = false
    private var scheduledBufferCount: Int = 0

    private var totalPendingCount: Int {
        criticalChunks.count + highChunks.count + normalChunks.count + lowChunks.count
    }

    private func popNextChunk() -> Data? {
        if !criticalChunks.isEmpty { return criticalChunks.removeFirst() }
        if !highChunks.isEmpty { return highChunks.removeFirst() }
        if !normalChunks.isEmpty { return normalChunks.removeFirst() }
        if !lowChunks.isEmpty { return lowChunks.removeFirst() }
        return nil
    }

    private func clearAllPending(keepingCapacity: Bool = false) {
        criticalChunks.removeAll(keepingCapacity: keepingCapacity)
        highChunks.removeAll(keepingCapacity: keepingCapacity)
        normalChunks.removeAll(keepingCapacity: keepingCapacity)
        lowChunks.removeAll(keepingCapacity: keepingCapacity)
    }
    private var jitterKickoffWorkItem: DispatchWorkItem?
    /// Set when drain stops due to chunk starvation (not barge-in).
    /// Allows faster restart with fewer buffered chunks.
    private var wasStarved = false
    private var restartObserver: NSObjectProtocol?
    private var suppressIncomingAudioUntil: CFAbsoluteTime = 0

    func setup() {
        // Gemini outputs PCM 24kHz mono 16-bit
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: SightLineConfig.audioOutputSampleRate,
            channels: 1,
            interleaved: true
        ) else {
            Self.logger.error("Failed to create playback audio format")
            return
        }
        playbackFormat = format

        // Ensure the shared player is ready
        if let player = SharedAudioEngine.shared.playerNode, !player.isPlaying {
            player.play()
        }

        schedulingQueue.async { [weak self] in
            guard let self = self else { return }
            self.clearAllPending()
            self.isDrainActive = false
            self.scheduledBufferCount = 0
            self.wasStarved = false
            self.jitterKickoffWorkItem?.cancel()
            self.jitterKickoffWorkItem = nil
        }

        // Clean up any existing observer before re-registering (prevents accumulation on repeated setup())
        if let obs = restartObserver {
            NotificationCenter.default.removeObserver(obs)
            restartObserver = nil
        }

        // Reset drain state if the shared engine restarts
        restartObserver = NotificationCenter.default.addObserver(
            forName: .sharedAudioEngineDidRestart,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self = self else { return }
            Self.logger.info("SharedAudioEngine restarted — resetting playback drain state")
            self.schedulingQueue.async {
                self.clearAllPending()
                self.isDrainActive = false
                self.scheduledBufferCount = 0
                self.wasStarved = false
                self.jitterKickoffWorkItem?.cancel()
                self.jitterKickoffWorkItem = nil
            }
            DispatchQueue.main.async { self.isPlaying = false }
        }

        DispatchQueue.main.async { self.isPlaying = false }
        Self.logger.info("Audio playback ready (shared engine)")
    }

    func playAudioData(_ data: Data) {
        playAudioData(data, priority: .normal)
    }

    func playAudioData(_ data: Data, priority: AudioPriority) {
        guard !data.isEmpty else { return }
        guard let format = playbackFormat,
              let player = SharedAudioEngine.shared.playerNode else { return }

        schedulingQueue.async { [weak self] in
            guard let self = self else { return }

            if CFAbsoluteTimeGetCurrent() < self.suppressIncomingAudioUntil {
                return
            }

            // CRITICAL priority: interrupt everything else
            if priority == .critical {
                self.interruptForCriticalAudio(player: player, format: format)
                self.criticalChunks.append(data)
                self.startDrain(player: player, format: format)
                return
            }

            // LOW priority: drop if total pending > 75% capacity
            if priority == .low && self.totalPendingCount >= (SightLineConfig.audioMaxPendingChunks * 3 / 4) {
                Self.logger.debug("Dropping low-priority audio chunk (buffer \(self.totalPendingCount)/\(SightLineConfig.audioMaxPendingChunks))")
                return
            }

            // Overflow guard: drop low first, then normal, then high. Never drop critical.
            if self.totalPendingCount >= SightLineConfig.audioMaxPendingChunks {
                let overflowCount = self.totalPendingCount
                Self.logger.warning("Audio buffer overflow (\(overflowCount) chunks), clearing for clean restart")
                self.lowChunks.removeAll(keepingCapacity: true)
                self.normalChunks.removeAll(keepingCapacity: true)
                self.highChunks.removeAll(keepingCapacity: true)
                // Keep criticalChunks
                self.isDrainActive = false
                self.scheduledBufferCount = 0
                self.wasStarved = true
                self.jitterKickoffWorkItem?.cancel()
                self.jitterKickoffWorkItem = nil
                let overflowCallback = self.onBufferOverflow
                DispatchQueue.main.async {
                    overflowCallback?(overflowCount)
                    NotificationCenter.default.post(name: .audioBufferOverflow, object: nil)
                }
            }

            // Append to appropriate priority bucket
            switch priority {
            case .critical: self.criticalChunks.append(data)
            case .high: self.highChunks.append(data)
            case .normal: self.normalChunks.append(data)
            case .low: self.lowChunks.append(data)
            }

            if self.isDrainActive {
                self.drainPendingChunks(player: player, format: format)
            } else if self.wasStarved {
                self.wasStarved = false
                self.startDrain(player: player, format: format)
            } else if self.totalPendingCount >= SightLineConfig.audioJitterBufferChunks {
                self.startDrain(player: player, format: format)
            } else {
                self.scheduleDrainFallback(player: player, format: format)
            }
        }
    }

    /// Temporarily drop newly received model audio chunks.
    /// Used after local barge-in to prevent stale pre-interrupt chunks from re-entering playback.
    func suppressIncomingAudio(for duration: TimeInterval) {
        schedulingQueue.async { [weak self] in
            guard let self = self else { return }
            let until = CFAbsoluteTimeGetCurrent() + max(0, duration)
            self.suppressIncomingAudioUntil = max(self.suppressIncomingAudioUntil, until)
            self.clearAllPending(keepingCapacity: true)
            self.wasStarved = false
        }
    }

    /// Stop playback immediately for barge-in support.
    /// Clears playback state asynchronously to avoid blocking the caller thread.
    func stopImmediately() {
        guard let player = SharedAudioEngine.shared.playerNode else { return }
        DispatchQueue.main.async { self.isPlaying = false }
        schedulingQueue.async { [weak self] in
            guard let self = self else { return }
            player.stop()
            player.reset()
            self.clearAllPending(keepingCapacity: true)
            self.isDrainActive = false
            self.scheduledBufferCount = 0
            self.wasStarved = false
            self.suppressIncomingAudioUntil = 0
            self.jitterKickoffWorkItem?.cancel()
            self.jitterKickoffWorkItem = nil
            player.play()
        }
    }

    func teardown() {
        schedulingQueue.async { [weak self] in
            guard let self = self else { return }
            self.clearAllPending()
            self.isDrainActive = false
            self.scheduledBufferCount = 0
            self.wasStarved = false
            self.suppressIncomingAudioUntil = 0
            self.jitterKickoffWorkItem?.cancel()
            self.jitterKickoffWorkItem = nil
        }

        if let obs = restartObserver {
            NotificationCenter.default.removeObserver(obs)
            restartObserver = nil
        }

        // Do NOT stop the shared engine — capture may still be active.
        // Just clear our own state.
        playbackFormat = nil
        DispatchQueue.main.async { self.isPlaying = false }
        Self.logger.info("Audio playback torn down (shared engine intact)")
    }

    /// Interrupt all non-critical audio for safety-critical playback.
    /// Follows the same stop+reset+play pattern as stopImmediately() (battle-tested by barge-in).
    private func interruptForCriticalAudio(player: AVAudioPlayerNode, format: AVAudioFormat) {
        player.stop()
        player.reset()
        normalChunks.removeAll(keepingCapacity: true)
        lowChunks.removeAll(keepingCapacity: true)
        highChunks.removeAll(keepingCapacity: true)
        // Keep criticalChunks
        scheduledBufferCount = 0
        isDrainActive = false
        wasStarved = false
        jitterKickoffWorkItem?.cancel()
        jitterKickoffWorkItem = nil
        player.play()
    }

    /// Fill the sliding window: schedule up to `audioScheduleAheadCount` buffers
    /// to AVAudioPlayerNode so there is always a next buffer queued.
    private func drainPendingChunks(player: AVAudioPlayerNode, format: AVAudioFormat) {
        guard isDrainActive else { return }

        while scheduledBufferCount < SightLineConfig.audioScheduleAheadCount,
              let chunk = popNextChunk() {
            guard let buffer = makePCMBuffer(from: chunk, format: format) else { continue }

            scheduledBufferCount += 1
            player.scheduleBuffer(buffer) { [weak self] in
                guard let self = self else { return }
                self.schedulingQueue.async {
                    self.scheduledBufferCount -= 1
                    guard self.isDrainActive else {
                        if self.scheduledBufferCount <= 0 {
                            self.scheduledBufferCount = 0
                            self.isDrainActive = false
                            DispatchQueue.main.async { self.isPlaying = false }
                        }
                        return
                    }
                    self.drainPendingChunks(player: player, format: format)
                }
            }
        }

        // All buffers drained and nothing left to play — mark starved so next
        // incoming chunk restarts drain immediately without waiting for full jitter buffer.
        if scheduledBufferCount == 0 && totalPendingCount == 0 {
            isDrainActive = false
            wasStarved = true
            let drainedCallback = self.onPlaybackDrained
            DispatchQueue.main.async {
                self.isPlaying = false
                drainedCallback?()
            }
        }
    }

    private func startDrain(player: AVAudioPlayerNode, format: AVAudioFormat) {
        jitterKickoffWorkItem?.cancel()
        jitterKickoffWorkItem = nil
        isDrainActive = true
        DispatchQueue.main.async { self.isPlaying = true }
        if !player.isPlaying {
            player.play()
        }
        drainPendingChunks(player: player, format: format)
    }

    private func scheduleDrainFallback(player: AVAudioPlayerNode, format: AVAudioFormat) {
        jitterKickoffWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            guard let self = self else { return }
            guard !self.isDrainActive else { return }
            guard self.totalPendingCount > 0 else { return }
            self.startDrain(player: player, format: format)
        }
        jitterKickoffWorkItem = workItem
        schedulingQueue.asyncAfter(
            deadline: .now() + SightLineConfig.audioJitterMaxWait,
            execute: workItem
        )
    }

    private func makePCMBuffer(from data: Data, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frameCount = UInt32(data.count / 2)  // 16-bit = 2 bytes per frame
        guard frameCount > 0 else { return nil }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else { return nil }

        buffer.frameLength = frameCount
        data.withUnsafeBytes { rawBufferPointer in
            if let baseAddress = rawBufferPointer.baseAddress,
               let channelData = buffer.int16ChannelData {
                memcpy(channelData[0], baseAddress, data.count)
            }
        }
        return buffer
    }
}
