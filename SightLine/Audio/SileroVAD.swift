//
//  SileroVAD.swift
//  SightLine
//
//  Client-side Voice Activity Detection using Silero VAD v4 (ONNX).
//  Processes 512-sample windows (32ms @ 16kHz) and provides speech onset/offset
//  callbacks with debouncing to avoid false triggers from ambient noise.
//
//  Singleton pattern matches SharedAudioEngine.shared convention.
//

import Foundation
import os

#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#endif

final class SileroVAD {
    static let shared = SileroVAD()

    private static let logger = Logger(subsystem: "com.sightline.app", category: "SileroVAD")

    // MARK: - Public State

    /// Whether speech is currently detected (after debounce).
    private(set) var isSpeechActive = false

    /// Latest raw probability from the model (0.0–1.0).
    private(set) var lastProbability: Float = 0

    // MARK: - Callbacks

    var onSpeechStart: (() -> Void)?
    var onSpeechEnd: (() -> Void)?
    var onProbabilityUpdate: ((Float) -> Void)?
    var onVADError: ((AudioPipelineError) -> Void)?

    // MARK: - Configuration

    /// Probability threshold for speech detection.
    private let threshold: Float = 0.5

    /// Number of consecutive speech frames required before onset (2 frames = 64ms).
    private let onsetFrames: Int = 2

    /// Number of consecutive non-speech frames required before offset (8 frames = 256ms).
    private let offsetFrames: Int = 8

    /// Silero VAD v4 expects 512 samples per window at 16kHz.
    private let windowSize: Int = 512

    /// Sample rate for Silero VAD.
    private let sampleRate: Int = 16000

    // MARK: - Internal State

    private var isModelLoaded = false
    private var loadAttempts: Int = 0
    private let maxLoadAttempts: Int = 3

    #if canImport(OnnxRuntimeBindings)
    private var session: ORTSession?
    private var env: ORTEnv?
    #endif

    /// LSTM state — shape [2, 1, 128] for Silero v4.
    private var lstmState: [Float] = []
    private let lstmStateSize = 2 * 1 * 128  // 256 floats

    /// Consecutive frames above/below threshold for debounce.
    private var speechFrameCount: Int = 0
    private var silenceFrameCount: Int = 0

    /// Accumulation buffer for partial audio chunks.
    private var sampleBuffer: [Float] = []

    /// Serial queue for thread-safe processing.
    private let processingQueue = DispatchQueue(
        label: "com.sightline.silero-vad",
        qos: .userInteractive
    )

    private init() {
        lstmState = [Float](repeating: 0, count: lstmStateSize)
    }

    // MARK: - Lifecycle

    func loadModel() {
        processingQueue.sync {
            guard !isModelLoaded else { return }
            guard loadAttempts < maxLoadAttempts else {
                Self.logger.warning("VAD model load skipped — max attempts (\(self.maxLoadAttempts)) reached")
                return
            }

            loadAttempts += 1

            #if canImport(OnnxRuntimeBindings)
            guard let modelPath = Bundle.main.path(forResource: "silero_vad", ofType: "onnx") else {
                Self.logger.error("silero_vad.onnx not found in bundle (attempt \(self.loadAttempts)/\(self.maxLoadAttempts))")
                let error: AudioPipelineError = .vadModelNotFound
                DispatchQueue.main.async { [weak self] in self?.onVADError?(error) }
                return
            }

            do {
                env = try ORTEnv(loggingLevel: .warning)
                let sessionOptions = try ORTSessionOptions()
                try sessionOptions.setIntraOpNumThreads(1)
                try sessionOptions.setGraphOptimizationLevel(.all)
                session = try ORTSession(env: env!, modelPath: modelPath, sessionOptions: sessionOptions)
                isModelLoaded = true
                Self.logger.info("Silero VAD model loaded successfully (attempt \(self.loadAttempts))")
            } catch {
                Self.logger.error("Failed to load Silero VAD model (attempt \(self.loadAttempts)/\(self.maxLoadAttempts)): \(error)")
                let pipelineError: AudioPipelineError = .vadModelLoadFailed
                DispatchQueue.main.async { [weak self] in self?.onVADError?(pipelineError) }
            }
            #else
            Self.logger.warning("onnxruntime not available — VAD disabled")
            let error: AudioPipelineError = .vadUnavailable
            DispatchQueue.main.async { [weak self] in self?.onVADError?(error) }
            #endif
        }
    }

    func reset() {
        processingQueue.sync {
            lstmState = [Float](repeating: 0, count: lstmStateSize)
            speechFrameCount = 0
            silenceFrameCount = 0
            isSpeechActive = false
            lastProbability = 0
            sampleBuffer.removeAll()
        }
    }

    // MARK: - Audio Processing

    /// Process raw PCM Int16 audio samples from AudioCaptureManager.
    /// Accumulates samples and runs inference on each complete 512-sample window.
    func processAudioFrame(_ samples: UnsafePointer<Int16>, count: Int) {
        // Early bail: skip float conversion entirely when model unavailable
        guard isModelLoaded else { return }

        // Convert Int16 to Float32 (normalized to [-1, 1])
        var floatSamples = [Float](repeating: 0, count: count)
        for i in 0..<count {
            floatSamples[i] = Float(samples[i]) / 32768.0
        }

        processingQueue.async { [weak self] in
            guard let self = self, self.isModelLoaded else { return }
            self.sampleBuffer.append(contentsOf: floatSamples)

            // Process all complete windows
            while self.sampleBuffer.count >= self.windowSize {
                let window = Array(self.sampleBuffer.prefix(self.windowSize))
                self.sampleBuffer.removeFirst(self.windowSize)
                self.runInference(window: window)
            }
        }
    }

    // MARK: - Inference

    private func runInference(window: [Float]) {
        #if canImport(OnnxRuntimeBindings)
        guard let session = session else { return }

        do {
            // Input tensor: [1, 512] float32
            let inputData = Data(bytes: window, count: window.count * MemoryLayout<Float>.size)
            let inputTensor = try ORTValue(
                tensorData: NSMutableData(data: inputData),
                elementType: .float,
                shape: [1, NSNumber(value: windowSize)]
            )

            // Sample rate tensor: [1] int64
            var sr = Int64(sampleRate)
            let srData = Data(bytes: &sr, count: MemoryLayout<Int64>.size)
            let srTensor = try ORTValue(
                tensorData: NSMutableData(data: srData),
                elementType: .int64,
                shape: [1]
            )

            // LSTM state: [2, 1, 128] float32 for Silero v4
            let stateData = Data(bytes: lstmState, count: lstmState.count * MemoryLayout<Float>.size)
            let stateTensor = try ORTValue(
                tensorData: NSMutableData(data: stateData),
                elementType: .float,
                shape: [2, 1, 128]
            )

            let outputs = try session.run(
                withInputs: [
                    "input": inputTensor,
                    "sr": srTensor,
                    "state": stateTensor,
                ],
                outputNames: ["output", "stateN"],
                runOptions: nil
            )

            // Extract probability
            guard let outputValue = outputs["output"] else { return }
            let outputData = try outputValue.tensorData() as Data
            let probability = outputData.withUnsafeBytes { $0.load(as: Float.self) }

            // Update LSTM state
            if let stateNValue = outputs["stateN"] {
                let stateNData = try stateNValue.tensorData() as Data
                stateNData.withUnsafeBytes { ptr in
                    let floats = ptr.bindMemory(to: Float.self)
                    for i in 0..<min(lstmStateSize, floats.count) {
                        lstmState[i] = floats[i]
                    }
                }
            }

            lastProbability = probability
            DispatchQueue.main.async { [weak self] in
                self?.onProbabilityUpdate?(probability)
            }

            updateSpeechState(probability: probability)

        } catch {
            Self.logger.error("Silero VAD inference error: \(error)")
        }
        #endif
    }

    // MARK: - Debounce Logic

    private func updateSpeechState(probability: Float) {
        let isSpeechFrame = probability >= threshold

        if isSpeechFrame {
            speechFrameCount += 1
            silenceFrameCount = 0
        } else {
            silenceFrameCount += 1
            speechFrameCount = 0
        }

        if !isSpeechActive && speechFrameCount >= onsetFrames {
            isSpeechActive = true
            Self.logger.debug("Speech onset detected (prob=\(probability, format: .fixed(precision: 3)))")
            DispatchQueue.main.async { [weak self] in
                self?.onSpeechStart?()
            }
        } else if isSpeechActive && silenceFrameCount >= offsetFrames {
            isSpeechActive = false
            Self.logger.debug("Speech offset detected (prob=\(probability, format: .fixed(precision: 3)))")
            DispatchQueue.main.async { [weak self] in
                self?.onSpeechEnd?()
            }
        }
    }
}
