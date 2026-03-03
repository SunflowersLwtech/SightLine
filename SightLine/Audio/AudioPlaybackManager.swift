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
    private var pendingChunks: [Data] = []
    private var isDrainActive = false
    private var scheduledBufferCount: Int = 0
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
            self.pendingChunks.removeAll()
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
                self.pendingChunks.removeAll()
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
        guard !data.isEmpty else { return }
        guard let format = playbackFormat,
              let player = SharedAudioEngine.shared.playerNode else { return }

        schedulingQueue.async { [weak self] in
            guard let self = self else { return }

            if CFAbsoluteTimeGetCurrent() < self.suppressIncomingAudioUntil {
                return
            }

            // Overflow guard: clear all pending + mark starved for clean restart.
            // Previous "drop oldest" strategy caused mid-sentence jumps that
            // sounded like the agent was talking over itself.
            if self.pendingChunks.count >= SightLineConfig.audioMaxPendingChunks {
                let overflowCount = self.pendingChunks.count
                Self.logger.warning("Audio buffer overflow (\(overflowCount) chunks), clearing for clean restart")
                self.pendingChunks.removeAll(keepingCapacity: true)
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

            self.pendingChunks.append(data)

            if self.isDrainActive {
                // Feed the sliding window immediately — avoids underrun recovery delay
                self.drainPendingChunks(player: player, format: format)
            } else if self.wasStarved {
                // Previous drain starved — restart immediately with whatever we have
                self.wasStarved = false
                self.startDrain(player: player, format: format)
            } else if self.pendingChunks.count >= SightLineConfig.audioJitterBufferChunks {
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
            self.pendingChunks.removeAll(keepingCapacity: true)
            self.wasStarved = false
        }
    }

    /// Stop playback immediately for barge-in support.
    /// Uses sync dispatch so state is fully cleared before player.play() restarts.
    func stopImmediately() {
        guard let player = SharedAudioEngine.shared.playerNode else { return }
        player.stop()
        player.reset()

        schedulingQueue.sync { [weak self] in
            guard let self = self else { return }
            self.pendingChunks.removeAll(keepingCapacity: true)
            self.isDrainActive = false
            self.scheduledBufferCount = 0
            self.wasStarved = false
            self.suppressIncomingAudioUntil = 0
            self.jitterKickoffWorkItem?.cancel()
            self.jitterKickoffWorkItem = nil
        }

        // Re-ready for next utterance after state is fully cleared
        player.play()
        DispatchQueue.main.async { self.isPlaying = false }
    }

    func teardown() {
        schedulingQueue.async { [weak self] in
            guard let self = self else { return }
            self.pendingChunks.removeAll()
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

    /// Fill the sliding window: schedule up to `audioScheduleAheadCount` buffers
    /// to AVAudioPlayerNode so there is always a next buffer queued.
    private func drainPendingChunks(player: AVAudioPlayerNode, format: AVAudioFormat) {
        guard isDrainActive else { return }

        while scheduledBufferCount < SightLineConfig.audioScheduleAheadCount,
              !pendingChunks.isEmpty {
            let chunk = pendingChunks.removeFirst()
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
        if scheduledBufferCount == 0 && pendingChunks.isEmpty {
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
            guard !self.pendingChunks.isEmpty else { return }
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
