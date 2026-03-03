//
//  FrameSelector.swift
//  SightLine
//
//  LOD-based frame throttling. Controls how often camera frames are sent
//  to the server based on the current Level of Detail:
//    LOD 1: 1 FPS (safety/navigation focus)
//    LOD 2: 1 FPS (balanced)
//    LOD 3: 0.5 FPS (static/low-activity scenes)
//

import Foundation
import Combine
import CoreGraphics
import ImageIO

class FrameSelector: ObservableObject {
    @Published var currentLOD: Int = 2
    @Published var effectiveFPS: Double = 0

    private var lastFrameTime: Date = .distantPast
    private var frameTimes: [Date] = []

    // Pixel-diff deduplication (SL-75)
    private var previousThumbnail: [UInt8]?
    private var lastSentTime: Date = .distantPast
    private let thumbnailSize = 32
    private let diffThreshold: Float = 5.0  // 0-255 scale; below this = "same scene"
    private let maxSkipDuration: TimeInterval = 5.0  // Force-send even static frames every 5s

    var minInterval: TimeInterval {
        switch currentLOD {
        case 1: return 1.0   // 1 FPS
        case 2: return 1.0   // 1 FPS
        case 3: return 2.0   // 0.5 FPS (static scenes)
        default: return 1.0
        }
    }

    func shouldSendFrame() -> Bool {
        let now = Date()
        return now.timeIntervalSince(lastFrameTime) >= minInterval
    }

    /// Check if the frame differs enough from the previous one to be worth sending.
    /// Returns true if the frame is sufficiently different or if no previous frame exists.
    ///
    /// IMPORTANT: Only updates the baseline thumbnail when the frame IS different.
    /// Previously, `defer` updated on every call — causing consecutive rejected frames
    /// (33ms apart at 30fps) to always compare nearly-identical thumbnails, permanently
    /// blocking all frames after the first rejection.
    ///
    /// Safety valve: force-sends at least one frame every `maxSkipDuration` seconds
    /// even if the scene is truly static, so the server always has fresh visual context.
    func isFrameDifferent(jpegData: Data) -> Bool {
        guard let thumbnail = downsampleToGrayscale(jpegData: jpegData) else {
            return true
        }

        guard let prev = previousThumbnail, prev.count == thumbnail.count else {
            previousThumbnail = thumbnail
            lastSentTime = Date()
            return true
        }

        // Force-send if we haven't sent anything for too long
        if Date().timeIntervalSince(lastSentTime) >= maxSkipDuration {
            previousThumbnail = thumbnail
            lastSentTime = Date()
            return true
        }

        let diff = meanAbsoluteDifference(prev, thumbnail)
        if diff >= diffThreshold {
            previousThumbnail = thumbnail
            lastSentTime = Date()
            return true
        }
        return false
    }

    func markFrameSent() {
        let now = Date()
        lastFrameTime = now
        frameTimes.append(now)
        // Keep only frames from the last 5 seconds for FPS calculation
        frameTimes = frameTimes.filter { now.timeIntervalSince($0) <= 5.0 }
        let elapsed = frameTimes.count > 1
            ? now.timeIntervalSince(frameTimes.first!)
            : 1.0
        DispatchQueue.main.async {
            self.effectiveFPS = elapsed > 0 ? Double(self.frameTimes.count - 1) / elapsed : 0
        }
    }

    /// Mark a frame as evaluated-but-skipped to keep throttle cadence stable.
    /// Without this, static scenes can trigger diff checks at capture FPS (30fps),
    /// creating noisy logs and unnecessary CPU usage.
    func markFrameSkipped() {
        lastFrameTime = Date()
    }

    func updateLOD(_ lod: Int) {
        DispatchQueue.main.async {
            self.currentLOD = lod
        }
    }

    // MARK: - Pixel-diff helpers

    private func downsampleToGrayscale(jpegData: Data) -> [UInt8]? {
        guard let source = CGImageSourceCreateWithData(jpegData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return nil
        }

        let width = thumbnailSize
        let height = thumbnailSize
        var pixels = [UInt8](repeating: 0, count: width * height)

        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixels
    }

    private func meanAbsoluteDifference(_ a: [UInt8], _ b: [UInt8]) -> Float {
        let count = min(a.count, b.count)
        guard count > 0 else { return Float.greatestFiniteMagnitude }

        var sum: Int = 0
        for i in 0..<count {
            sum += abs(Int(a[i]) - Int(b[i]))
        }
        return Float(sum) / Float(count)
    }
}
