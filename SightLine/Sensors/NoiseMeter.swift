//
//  NoiseMeter.swift
//  SightLine
//
//  Calculates ambient noise level (dB SPL) from RMS values provided
//  by AudioCaptureManager. Does NOT create its own AVAudioEngine.
//  Uses a sliding average of the last 5 readings for stability.
//

import Foundation
import Combine
import os

class NoiseMeter: ObservableObject {
    @Published var ambientNoiseDb: Double = 50.0
    /// True once at least one real RMS sample has been processed.
    private(set) var hasReceivedData: Bool = false

    private static let logger = Logger(subsystem: "com.sightline.app", category: "NoiseMeter")

    private var rmsHistory: [Float] = []
    private let historySize = 20
    private let calibrationOffset: Float = 100.0  // approximate dB SPL calibration

    /// Process a raw RMS value from the audio capture tap.
    /// Called by AudioCaptureManager's onAudioLevelUpdate callback.
    /// Uses 25th-percentile over a 20-sample window to filter speech spikes.
    func processRMS(_ rms: Float) {
        hasReceivedData = true
        rmsHistory.append(rms)
        if rmsHistory.count > historySize {
            rmsHistory.removeFirst()
        }

        let sorted = rmsHistory.sorted()
        let p25Rms = sorted[sorted.count / 4]
        let db = Double(20.0 * log10(max(p25Rms, 1e-6)) + calibrationOffset)
        let clampedDb = max(0.0, min(120.0, db))

        DispatchQueue.main.async {
            self.ambientNoiseDb = clampedDb
        }
    }

    /// Reset noise meter state.
    func reset() {
        rmsHistory.removeAll()
        hasReceivedData = false
        DispatchQueue.main.async {
            self.ambientNoiseDb = 50.0
        }
    }
}
