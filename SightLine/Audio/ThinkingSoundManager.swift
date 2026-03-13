//
//  ThinkingSoundManager.swift
//  SightLine
//
//  Procedurally generated thinking/processing sounds for blind user feedback.
//  Uses a separate AVAudioPlayerNode (ambientPlayerNode) at low volume
//  so it doesn't trigger barge-in echo gating.
//

import AVFoundation
import os

final class ThinkingSoundManager {
    enum SoundState: Equatable {
        case idle
        case thinking   // 440Hz, 2Hz AM pulse — general processing
        case searching  // 440+554Hz, 3Hz AM pulse — tool/search in progress
    }

    private static let logger = Logger(subsystem: "com.sightline.app", category: "ThinkingSound")

    private var currentState: SoundState = .idle
    private var thinkingBuffer: AVAudioPCMBuffer?
    private var searchingBuffer: AVAudioPCMBuffer?
    private let queue = DispatchQueue(label: "com.sightline.audio.thinking")

    func setup() {
        thinkingBuffer = generatePulsingTone(baseFreq: 440, modFreq: 2.0, duration: 0.5)
        searchingBuffer = generatePulsingTone(baseFreq: 440, modFreq: 3.0, duration: 0.5, secondFreq: 554)
        Self.logger.info("ThinkingSoundManager ready")
    }

    func setState(_ newState: SoundState) {
        queue.async { [weak self] in
            guard let self, newState != self.currentState else { return }
            self.stopCurrent()
            self.currentState = newState
            if newState != .idle,
               let buffer = (newState == .thinking ? self.thinkingBuffer : self.searchingBuffer) {
                SharedAudioEngine.shared.ambientPlayerNode?.scheduleBuffer(buffer, at: nil, options: .loops)
                SharedAudioEngine.shared.ambientPlayerNode?.play()
                Self.logger.debug("Thinking sound: \(String(describing: newState))")
            }
        }
    }

    func stopImmediately() {
        queue.async { [weak self] in
            self?.currentState = .idle
            self?.stopCurrent()
        }
    }

    private func stopCurrent() {
        let player = SharedAudioEngine.shared.ambientPlayerNode
        player?.stop()
        player?.reset()
        player?.play()
    }

    private func generatePulsingTone(
        baseFreq: Float,
        modFreq: Float,
        duration: Float,
        secondFreq: Float? = nil
    ) -> AVAudioPCMBuffer? {
        let sampleRate: Float = Float(SightLineConfig.audioOutputSampleRate)
        let frameCount = UInt32(sampleRate * duration)
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: true
        ), let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        for i in 0..<Int(frameCount) {
            let t = Float(i) / sampleRate
            var sample = sin(2 * .pi * baseFreq * t)
            if let f2 = secondFreq {
                sample += sin(2 * .pi * f2 * t) * 0.5
                sample *= 0.67
            }
            let envelope = 0.5 + 0.5 * sin(2 * .pi * modFreq * t) // AM modulation
            data[i] = sample * envelope * 0.3 // low amplitude
        }
        return buffer
    }

    func teardown() {
        stopImmediately()
    }
}
