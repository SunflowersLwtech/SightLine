//
//  CameraManager.swift
//  SightLine
//
//  Captures frames from the camera, resizes to 768x768,
//  encodes as JPEG, and delivers via callback for WebSocket transmission.
//  Supports front/back camera flip. Frame rate is controlled by FrameSelector.
//

import AVFoundation
import Combine
import UIKit
import CoreImage
import os

class CameraManager: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var previewSession: AVCaptureSession?
    @Published var cameraPosition: AVCaptureDevice.Position = .back

    private static let logger = Logger(subsystem: "com.sightline.app", category: "Camera")

    private var captureSession: AVCaptureSession?
    private let sessionQueue = DispatchQueue(label: "com.sightline.camera")
    private let context = CIContext()

    var onFrameCaptured: ((Data) -> Void)?  // JPEG data callback
    var onCameraFailure: ((String) -> Void)?
    var onDepthEstimated: ((DepthEstimator.DepthSummary) -> Void)?
    var frameSelector: FrameSelector?
    var depthEstimator: DepthEstimator?

    func startCapture() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.captureSession?.isRunning == true {
                return
            }
            self.setupCaptureSession()
        }
    }

    /// Flip between front and back camera while keeping the session alive.
    func flipCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self, let session = self.captureSession, session.isRunning else { return }

            let newPosition: AVCaptureDevice.Position = (self.cameraPosition == .back) ? .front : .back

            guard let newDevice = AVCaptureDevice.default(
                .builtInWideAngleCamera, for: .video, position: newPosition
            ), let newInput = try? AVCaptureDeviceInput(device: newDevice) else {
                Self.logger.error("Camera not available for position \(newPosition.rawValue)")
                return
            }

            session.beginConfiguration()
            // Remove existing video input
            if let currentInput = session.inputs.first as? AVCaptureDeviceInput {
                session.removeInput(currentInput)
            }
            if session.canAddInput(newInput) {
                session.addInput(newInput)
            }
            session.commitConfiguration()

            DispatchQueue.main.async {
                self.cameraPosition = newPosition
            }
            Self.logger.info("Camera flipped to \(newPosition == .back ? "back" : "front")")
        }
    }

    private func setupCaptureSession() {
        let session = AVCaptureSession()
        session.sessionPreset = .medium

        let position = cameraPosition

        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: position
        ) else {
            Self.logger.error("Camera not available for position \(position.rawValue)")
            onCameraFailure?("camera_not_available")
            return
        }

        guard let input = try? AVCaptureDeviceInput(device: camera) else {
            Self.logger.error("Failed to create camera input")
            onCameraFailure?("camera_input_creation_failed")
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: sessionQueue)
        output.alwaysDiscardsLateVideoFrames = true

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.startRunning()
        captureSession = session

        DispatchQueue.main.async {
            self.isRunning = true
            self.previewSession = session
        }
        Self.logger.info("Camera capture started (position: \(position == .back ? "back" : "front"))")
    }

    func stopCapture() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.captureSession?.stopRunning()
            self.captureSession = nil
            DispatchQueue.main.async {
                self.isRunning = false
                self.previewSession = nil
            }
            Self.logger.info("Camera capture stopped")
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Check with frame selector for LOD-based throttling
        if let selector = frameSelector, !selector.shouldSendFrame() {
            return
        }

        // Run depth estimation on accepted frames (same cadence as vision)
        if let estimator = depthEstimator, estimator.isAvailable,
           let summary = estimator.estimateDepth(from: pixelBuffer) {
            onDepthEstimated?(summary)
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Resize to 768x768 maintaining aspect ratio
        let targetSize = CGSize(
            width: SightLineConfig.videoFrameWidth,
            height: SightLineConfig.videoFrameHeight
        )
        let scaleX = targetSize.width / ciImage.extent.width
        let scaleY = targetSize.height / ciImage.extent.height
        let scale = min(scaleX, scaleY)
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        guard let cgImage = context.createCGImage(scaledImage, from: scaledImage.extent) else {
            return
        }

        let uiImage = UIImage(cgImage: cgImage)
        guard let jpegData = uiImage.jpegData(compressionQuality: SightLineConfig.jpegQuality) else {
            return
        }

        onFrameCaptured?(jpegData)
    }
}
