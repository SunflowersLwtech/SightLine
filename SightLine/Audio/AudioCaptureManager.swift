//
//  AudioCaptureManager.swift
//  SightLine
//
//  Captures microphone audio using the shared AVAudioEngine, converts to
//  PCM 16kHz mono 16-bit, and delivers raw PCM data via callback for
//  WebSocket transmission.
//
//  Uses SharedAudioEngine.shared.engine.inputNode so that capture and
//  playback share a single audio graph, enabling hardware AEC.
//
//  Threading model:
//    - All lifecycle mutations (start/stop/observer callbacks) serialized on `captureQueue`.
//    - Tap callback runs on Core Audio realtime thread; uses `tapGeneration` as
//      lock-free stale guard (UInt64 reads are atomic on arm64).
//    - `converter`/`targetFormat` set on `captureQueue` before `installTap`;
//      installTap acts as memory barrier.
//

import AVFoundation
import Combine
import os

class AudioCaptureManager: ObservableObject {
    @Published var isCapturing = false

    private static let logger = Logger(subsystem: "com.sightline.app", category: "AudioCapture")

    var onAudioCaptured: ((Data) -> Void)?
    /// RMS audio level callback for NoiseMeter (ambient noise calculation).
    var onAudioLevelUpdate: ((Float) -> Void)?
    /// Returns whether model audio is still actively playing on device.
    /// Helps prevent echo leakage when playback queue is draining after network burst.
    var isModelAudioPlaying: (() -> Bool)?

    /// Timestamp of the last model audio chunk received via WebSocket.
    /// Used with `modelSpeakingTimeout` to determine if the model is currently speaking.
    /// Timestamp-based approach is more robust than a boolean flag because it handles
    /// playback buffer starvation (network jitter) gracefully — no false toggles.
    var lastModelAudioReceivedAt: CFAbsoluteTime = 0

    /// How long after the last audio chunk we still consider the model "speaking".
    /// Reduced from 2.5s to 1.5s to shrink the dead zone after model finishes,
    /// giving the user more time to speak before the next context flush.
    private let modelSpeakingTimeout: Double = 1.5

    /// RMS threshold for voice barge-in during model playback.
    /// Post-AEC residual echo is typically < 0.02 RMS; human speech at arm's length > 0.08.
    private let bargeInRMSThreshold: Float = 0.12

    /// Called when voice barge-in detected (RMS above threshold during model speech).
    var onVoiceBargeIn: (() -> Void)?

    /// Last computed RMS value, reused by echo gating logic.
    private var lastRMS: Float = 0

    /// Barge-in confirmation counter: requires consecutive frames above threshold + VAD active.
    /// Prevents AEC residual echo bursts (< 200ms) from triggering false barge-in.
    private var bargeInConfirmCount: Int = 0
    private let bargeInConfirmRequired: Int = 6  // ~600ms at 100ms/frame — more robust against echo accumulation

    /// Ignore barge-in candidates immediately after each model audio chunk.
    /// This protects against AEC tail and speaker leakage right after playback resumes.
    private let bargeInGuardAfterModelChunkSec: Double = 0.15  // AEC processes in 10-30ms; 150ms sufficient

    /// Require stronger VAD confidence for barge-in while model is speaking.
    /// `isSpeechActive` alone can remain true for a short hangover window.
    private let bargeInVADProbabilityThreshold: Float = 0.75

    /// Echo suppression: count consecutive guard windows with RMS above threshold.
    /// If >= echoGuardWindowThreshold consecutive windows trigger, it's likely echo
    /// leaking through AEC rather than real user speech → suppress barge-in.
    private var echoGuardWindowCount: Int = 0
    private let echoGuardWindowThreshold: Int = 3

    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?
    private var restartObserver: NSObjectProtocol?
    private var pauseObserver: NSObjectProtocol?

    /// Serial queue serializing all lifecycle mutations (start/stop/observer callbacks).
    /// Matches AudioPlaybackManager's `schedulingQueue` pattern.
    private let captureQueue = DispatchQueue(label: "com.sightline.audio.capture", qos: .userInitiated)

    /// Generation counter invalidating stale tap callbacks after stop/restart.
    /// Incremented on `captureQueue` before each tap install; tap closure captures
    /// the current value and bails if it no longer matches.
    private var tapGeneration: UInt64 = 0

    // MARK: - Public API

    func startCapture() {
        captureQueue.async { [weak self] in
            self?._startCaptureOnQueue()
        }
    }

    func stopCapture() {
        captureQueue.async { [weak self] in
            self?._stopCaptureOnQueue()
        }
    }

    // MARK: - Lifecycle (captureQueue)

    private func _startCaptureOnQueue() {
        guard let engine = SharedAudioEngine.shared.engine, engine.isRunning else {
            Self.logger.error("SharedAudioEngine not running; cannot start capture")
            return
        }

        // Clean up any existing observers before re-registering (prevents accumulation)
        _removeObserversOnQueue()

        // Increment generation to invalidate any in-flight tap callback from a prior cycle
        tapGeneration &+= 1
        let currentGeneration = tapGeneration

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Target format: PCM 16kHz Mono 16-bit (what Gemini expects)
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: SightLineConfig.audioInputSampleRate,
            channels: 1,
            interleaved: true
        ) else {
            Self.logger.error("Failed to create target audio format")
            return
        }
        targetFormat = fmt

        guard let conv = AVAudioConverter(from: inputFormat, to: fmt) else {
            Self.logger.error("Failed to create audio converter (input: \(inputFormat))")
            return
        }
        converter = conv

        // Remove any existing tap before installing a new one
        _removeTapOnQueue()

        inputNode.installTap(onBus: 0, bufferSize: SightLineConfig.audioBufferSize, format: inputFormat) {
            [weak self] buffer, _ in
            guard let self = self else { return }
            // Stale guard: bail if a newer start/stop cycle has begun.
            // UInt64 reads are atomic on arm64, no lock needed.
            guard self.tapGeneration == currentGeneration else { return }
            guard let fmt = self.targetFormat, let converter = self.converter else { return }

            let targetFrameCount = AVAudioFrameCount(SightLineConfig.audioBufferSize)
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: fmt,
                frameCapacity: targetFrameCount
            ) else { return }

            var error: NSError?
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            if let error = error {
                Self.logger.error("Audio conversion error: \(error)")
                return
            }

            // Calculate RMS for NoiseMeter (from input buffer, not converted output)
            if let floatData = buffer.floatChannelData {
                let frameLength = Int(buffer.frameLength)
                if frameLength > 0 {
                    var sumSquares: Float = 0
                    let samples = floatData[0]
                    for i in 0..<frameLength {
                        sumSquares += samples[i] * samples[i]
                    }
                    let rms = sqrtf(sumSquares / Float(frameLength))
                    self.lastRMS = rms
                    self.onAudioLevelUpdate?(rms)
                }
            }

            if let channelData = outputBuffer.int16ChannelData {
                let byteCount = Int(outputBuffer.frameLength) * 2  // 16-bit = 2 bytes per sample

                let now = CFAbsoluteTimeGetCurrent()
                let playbackActive = self.isModelAudioPlaying?() ?? false
                let isModelCurrentlySpeaking =
                    playbackActive || (now - self.lastModelAudioReceivedAt) < self.modelSpeakingTimeout

                if isModelCurrentlySpeaking {
                    // Model speaking: send silence to prevent echo reaching Gemini VAD.
                    // Multi-frame confirmation prevents AEC residual bursts from false barge-in.
                    if (now - self.lastModelAudioReceivedAt) < self.bargeInGuardAfterModelChunkSec {
                        // Inside guard window: reset confirm count to prevent
                        // echo residuals from accumulating across guard windows.
                        // Track whether guard windows consistently have high RMS (echo pattern).
                        let rms = self.lastRMS
                        if rms > self.bargeInRMSThreshold {
                            self.echoGuardWindowCount += 1
                        } else {
                            self.echoGuardWindowCount = 0
                        }
                        self.bargeInConfirmCount = 0
                        let silence = Data(count: byteCount)
                        self.onAudioCaptured?(silence)
                        return
                    }

                    let rms = self.lastRMS
                    if rms > self.bargeInRMSThreshold {
                        // Echo suppression: if consecutive guard windows all had
                        // high RMS, this is likely speaker echo leaking through AEC,
                        // not real user speech → suppress barge-in entirely.
                        if self.echoGuardWindowCount >= self.echoGuardWindowThreshold {
                            Self.logger.debug("Barge-in suppressed: echo detected (\(self.echoGuardWindowCount) consecutive guard windows)")
                            self.bargeInConfirmCount = 0
                            let silence = Data(count: byteCount)
                            self.onAudioCaptured?(silence)
                            return
                        }

                        SileroVAD.shared.processAudioFrame(channelData[0], count: Int(outputBuffer.frameLength))
                        if SileroVAD.shared.isSpeechActive &&
                            SileroVAD.shared.lastProbability >= self.bargeInVADProbabilityThreshold {
                            self.bargeInConfirmCount += 1
                            if self.bargeInConfirmCount >= self.bargeInConfirmRequired {
                                // Sustained speech confirmed: real barge-in
                                self.bargeInConfirmCount = 0
                                self.echoGuardWindowCount = 0
                                let data = Data(bytes: channelData[0], count: byteCount)
                                self.onAudioCaptured?(data)
                                self.lastModelAudioReceivedAt = 0  // Exit speaking state
                                self.onVoiceBargeIn?()
                            } else {
                                // Still confirming: send silence while accumulating evidence
                                let silence = Data(count: byteCount)
                                self.onAudioCaptured?(silence)
                            }
                        } else {
                            // RMS high but VAD says no speech: reset counter, send silence
                            self.bargeInConfirmCount = 0
                            let silence = Data(count: byteCount)
                            self.onAudioCaptured?(silence)
                        }
                    } else {
                        // Echo residual (RMS < threshold): reset counters, send silence
                        self.bargeInConfirmCount = 0
                        self.echoGuardWindowCount = 0
                        let silence = Data(count: byteCount)
                        self.onAudioCaptured?(silence)
                    }
                } else {
                    // Model idle: reset all barge-in counters, send real audio
                    self.bargeInConfirmCount = 0
                    self.echoGuardWindowCount = 0
                    SileroVAD.shared.processAudioFrame(channelData[0], count: Int(outputBuffer.frameLength))
                    let data = Data(bytes: channelData[0], count: byteCount)
                    self.onAudioCaptured?(data)
                }
            }
        }

        // Register fresh observers
        _registerObserversOnQueue()

        // Initialize client-side VAD for speech detection
        SileroVAD.shared.loadModel()
        SileroVAD.shared.reset()
        SileroVAD.shared.onSpeechStart = nil
        SileroVAD.shared.onSpeechEnd = nil

        DispatchQueue.main.async { self.isCapturing = true }
        Self.logger.info("Audio capture started (shared engine, VP=\(SharedAudioEngine.shared.isVoiceProcessingEnabled))")
    }

    private func _stopCaptureOnQueue() {
        // Increment generation to invalidate any in-flight tap callbacks
        tapGeneration &+= 1

        _removeTapOnQueue()
        converter = nil
        targetFormat = nil

        _removeObserversOnQueue()

        DispatchQueue.main.async { self.isCapturing = false }
        Self.logger.info("Audio capture stopped")
    }

    // MARK: - Helpers (must be called on captureQueue)

    private func _removeTapOnQueue() {
        // Only remove tap — do NOT stop the shared engine (playback may still be active)
        SharedAudioEngine.shared.engine?.inputNode.removeTap(onBus: 0)
    }

    private func _removeObserversOnQueue() {
        if let obs = pauseObserver {
            NotificationCenter.default.removeObserver(obs)
            pauseObserver = nil
        }
        if let obs = restartObserver {
            NotificationCenter.default.removeObserver(obs)
            restartObserver = nil
        }
    }

    private func _registerObserversOnQueue() {
        // Remove tap cleanly when engine pauses (phone call / Siri interruption).
        // isCapturing stays true so the didRestart observer re-installs the tap.
        pauseObserver = NotificationCenter.default.addObserver(
            forName: .sharedAudioEngineDidPause,
            object: nil,
            queue: nil  // deliver on posting thread, then dispatch to captureQueue
        ) { [weak self] _ in
            guard let self = self else { return }
            self.captureQueue.async {
                guard self.isCapturing else { return }
                Self.logger.info("SharedAudioEngine paused — removing capture tap")
                self._removeTapOnQueue()
            }
        }

        // Re-install tap if the shared engine restarts (route change, interruption)
        restartObserver = NotificationCenter.default.addObserver(
            forName: .sharedAudioEngineDidRestart,
            object: nil,
            queue: nil  // deliver on posting thread, then dispatch to captureQueue
        ) { [weak self] _ in
            guard let self = self else { return }
            self.captureQueue.async {
                guard self.isCapturing else { return }
                Self.logger.info("SharedAudioEngine restarted — re-installing capture tap")
                self._startCaptureOnQueue()
            }
        }
    }
}
