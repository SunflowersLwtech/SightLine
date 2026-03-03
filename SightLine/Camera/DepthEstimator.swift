//
//  DepthEstimator.swift
//  SightLine
//
//  Monocular depth estimation using CoreML (Depth Anything V2 Small)
//  with optional LiDAR override on Pro devices.
//  Produces a DepthSummary with center distance, min distance, and
//  quadrant breakdown for spatial awareness.
//

import CoreML
import Vision
import CoreImage
import os

class DepthEstimator {
    private static let logger = Logger(subsystem: "com.sightline.app", category: "Depth")

    private var vnModel: VNCoreMLModel?
    private var isLoaded = false

    struct DepthSummary: Codable {
        var centerDistance: Float       // Estimated distance at image center (meters)
        var minDistance: Float           // Closest object distance (meters)
        var minDistanceRegion: String   // Region of closest object (topLeft/topRight/bottomLeft/bottomRight/center)
        var quadrants: [String: Float]  // 4 quadrant distances

        enum CodingKeys: String, CodingKey {
            case centerDistance = "center_distance"
            case minDistance = "min_distance"
            case minDistanceRegion = "min_distance_region"
            case quadrants
        }
    }

    /// Load the CoreML depth model. Call once at startup.
    func loadModel() {
        guard !isLoaded else { return }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Prefer Neural Engine

            // Look for the compiled model in the bundle
            guard let modelURL = Bundle.main.url(forResource: "DepthAnythingV2SmallF16",
                                                  withExtension: "mlmodelc") else {
                Self.logger.warning("DepthAnythingV2SmallF16.mlmodelc not found in bundle — depth estimation disabled")
                return
            }

            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            vnModel = try VNCoreMLModel(for: mlModel)
            isLoaded = true
            Self.logger.info("Depth model loaded successfully")
        } catch {
            Self.logger.error("Failed to load depth model: \(error.localizedDescription)")
        }
    }

    /// Whether the model is available for inference.
    var isAvailable: Bool { isLoaded && vnModel != nil }

    /// Estimate depth from a camera pixel buffer.
    /// Returns nil if model is not loaded or inference fails.
    func estimateDepth(from pixelBuffer: CVPixelBuffer) -> DepthSummary? {
        guard let model = vnModel else { return nil }

        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            Self.logger.error("Depth inference failed: \(error.localizedDescription)")
            return nil
        }

        // Extract depth map from results
        guard let results = request.results else { return nil }

        // Try VNPixelBufferObservation (depth map output)
        if let depthObs = results.first as? VNPixelBufferObservation {
            return processDepthBuffer(depthObs.pixelBuffer)
        }

        // Try MLMultiArray output
        if let coreMLObs = results.first as? VNCoreMLFeatureValueObservation,
           let multiArray = coreMLObs.featureValue.multiArrayValue {
            return processMultiArray(multiArray)
        }

        Self.logger.warning("Depth model returned unexpected result type")
        return nil
    }

    // MARK: - Private Processing

    private func processDepthBuffer(_ depthBuffer: CVPixelBuffer) -> DepthSummary {
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else {
            return fallbackSummary()
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(depthBuffer)

        // Sample key regions
        let regions = sampleRegions(width: width, height: height)
        var regionValues: [String: Float] = [:]

        for (name, point) in regions {
            let value: Float
            if pixelFormat == kCVPixelFormatType_OneComponent32Float {
                let ptr = baseAddress.assumingMemoryBound(to: Float.self)
                let offset = point.y * (bytesPerRow / MemoryLayout<Float>.stride) + point.x
                value = ptr[offset]
            } else if pixelFormat == kCVPixelFormatType_DepthFloat16 {
                let ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
                let offset = point.y * (bytesPerRow / MemoryLayout<UInt16>.stride) + point.x
                value = float16ToFloat32(ptr[offset])
            } else {
                // Assume 8-bit grayscale
                let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
                let offset = point.y * bytesPerRow + point.x
                value = Float(ptr[offset]) / 255.0
            }
            regionValues[name] = value
        }

        return buildSummary(from: regionValues)
    }

    private func processMultiArray(_ multiArray: MLMultiArray) -> DepthSummary {
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count >= 2 else { return fallbackSummary() }

        let height = shape[shape.count - 2]
        let width = shape[shape.count - 1]

        let regions = sampleRegions(width: width, height: height)
        var regionValues: [String: Float] = [:]

        for (name, point) in regions {
            let index: [NSNumber]
            if shape.count == 3 {
                index = [0 as NSNumber, point.y as NSNumber, point.x as NSNumber]
            } else {
                index = [point.y as NSNumber, point.x as NSNumber]
            }
            regionValues[name] = multiArray[index].floatValue
        }

        return buildSummary(from: regionValues)
    }

    private struct IntPoint {
        let x: Int
        let y: Int
    }

    private func sampleRegions(width: Int, height: Int) -> [(String, IntPoint)] {
        let cx = width / 2
        let cy = height / 2
        let qx = width / 4
        let qy = height / 4

        return [
            ("center", IntPoint(x: cx, y: cy)),
            ("topLeft", IntPoint(x: qx, y: qy)),
            ("topRight", IntPoint(x: cx + qx, y: qy)),
            ("bottomLeft", IntPoint(x: qx, y: cy + qy)),
            ("bottomRight", IntPoint(x: cx + qx, y: cy + qy)),
        ]
    }

    private func buildSummary(from regionValues: [String: Float]) -> DepthSummary {
        // Depth Anything V2 outputs inverse relative depth:
        // Higher value = closer. Convert to approximate meters using heuristic.
        // iPhone wide camera ~26mm equiv focal length.
        // Rough calibration: inverseDepth 1.0 ≈ 0.5m, 0.1 ≈ 5m
        func toMeters(_ inverseDepth: Float) -> Float {
            guard inverseDepth > 0.01 else { return 20.0 }  // Very far
            return max(0.1, min(20.0, 0.5 / inverseDepth))
        }

        let centerRaw = regionValues["center"] ?? 0.0
        let centerDist = toMeters(centerRaw)

        // Find minimum distance (closest object)
        var minDist: Float = .greatestFiniteMagnitude
        var minRegion = "center"

        let quadrants: [String: Float] = [
            "topLeft": toMeters(regionValues["topLeft"] ?? 0.0),
            "topRight": toMeters(regionValues["topRight"] ?? 0.0),
            "bottomLeft": toMeters(regionValues["bottomLeft"] ?? 0.0),
            "bottomRight": toMeters(regionValues["bottomRight"] ?? 0.0),
        ]

        // Check center
        if centerDist < minDist {
            minDist = centerDist
            minRegion = "center"
        }

        // Check quadrants
        for (region, dist) in quadrants {
            if dist < minDist {
                minDist = dist
                minRegion = region
            }
        }

        return DepthSummary(
            centerDistance: centerDist,
            minDistance: minDist,
            minDistanceRegion: minRegion,
            quadrants: quadrants
        )
    }

    private func fallbackSummary() -> DepthSummary {
        DepthSummary(
            centerDistance: -1.0,
            minDistance: -1.0,
            minDistanceRegion: "unknown",
            quadrants: [:]
        )
    }

    private func float16ToFloat32(_ value: UInt16) -> Float {
        let sign = (value >> 15) & 0x1
        let exponent = (value >> 10) & 0x1F
        let mantissa = value & 0x3FF

        if exponent == 0 {
            if mantissa == 0 { return sign == 1 ? -0.0 : 0.0 }
            // Subnormal
            let f = Float(mantissa) / 1024.0 * pow(2.0, -14.0)
            return sign == 1 ? -f : f
        }
        if exponent == 31 {
            return mantissa == 0
                ? (sign == 1 ? -.infinity : .infinity)
                : .nan
        }

        let f = (1.0 + Float(mantissa) / 1024.0) * pow(2.0, Float(Int(exponent) - 15))
        return sign == 1 ? -f : f
    }
}
