//
//  DebugOverlay.swift
//  SightLine
//
//  Debug data model (DebugOverlayModel) and camera preview UIViewRepresentable.
//  DebugOverlayModel is the data hub consumed by DeveloperConsoleModel.
//  CameraPreviewView is used by both MainView and DeveloperConsoleView.
//

import SwiftUI
import AVFoundation
import Combine
import os

private let logger = Logger(subsystem: "com.sightline.app", category: "DebugOverlay")

// MARK: - Debug Data Model

@MainActor
final class DebugOverlayModel: ObservableObject {
    @Published var currentLOD: Int = 2
    @Published var previousLOD: Int = 2
    @Published var lodReason: String = ""
    @Published var triggeredRules: [String] = []

    // Telemetry snapshot
    @Published var motionState: String = "unknown"
    @Published var heartRate: Double?
    @Published var noiseDb: Double = 50.0
    @Published var stepCadence: Double = 0.0

    // Connection
    @Published var isConnected: Bool = false
    @Published var isSafeMode: Bool = false
    @Published var activityState: String = "idle"
    @Published var activityEvent: String = "--"
    @Published var activityQueueStatus: String = "--"
    @Published var activityTimestamp: String = "--"

    // Sub-agent capabilities
    @Published var visionStatus: String = "ready"
    @Published var ocrStatus: String = "ready"
    @Published var faceStatus: String = "ready"

    // Memory Top 3 (SL-77)
    @Published var memoryTop3: [String] = []
    @Published var memoryTop3Detailed: [[String: Any]] = []

    // GPS
    @Published var latitude: Double = 0
    @Published var longitude: Double = 0

    // Frame rate
    @Published var frameRate: Double = 0

    // Latency
    @Published var lastEventTime: Date?

    var latencyText: String {
        guard let t = lastEventTime else { return "--" }
        let ms = Int(Date().timeIntervalSince(t) * 1000)
        return "\(ms)ms"
    }

    func updateFromLodDebug(_ data: [String: Any]) {
        if let lod = data["lod"] as? Int { currentLOD = lod }
        if let prev = data["prev"] as? Int { previousLOD = prev }
        if let reason = data["reason"] as? String { lodReason = reason }
        if let rules = data["rules"] as? [String] { triggeredRules = rules }
        if let motion = data["motion"] as? String { motionState = motion }
        if let hr = data["hr"] as? Double { heartRate = hr }
        if let noise = data["noise_db"] as? Double { noiseDb = noise }
        if let cadence = data["cadence"] as? Double { stepCadence = cadence }
        if let memories = data["memory_top3"] as? [String] {
            memoryTop3 = Array(memories.prefix(3))
        }
        if let detailed = data["memory_top3_detailed"] as? [[String: Any]] {
            memoryTop3Detailed = detailed
        }
        lastEventTime = Date()
    }

    func markCapabilityDegraded(_ capability: String) {
        switch capability {
        case "vision": visionStatus = "degraded"
        case "ocr": ocrStatus = "degraded"
        case "face": faceStatus = "degraded"
        default: break
        }
    }

    func markCapabilityReady(_ capability: String) {
        switch capability {
        case "vision": visionStatus = "ready"
        case "ocr": ocrStatus = "ready"
        case "face": faceStatus = "ready"
        default: break
        }
    }

    func updateFromActivityDebug(_ data: [String: Any]) {
        if let state = data["state"] as? String { activityState = state }
        if let event = data["event"] as? String { activityEvent = event }
        if let status = data["queue_status"] as? String { activityQueueStatus = status }
        if let ts = data["timestamp"] as? String { activityTimestamp = ts }
        lastEventTime = Date()
    }
}

// MARK: - Camera Preview (UIViewRepresentable)

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewContainerView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            // Safe by construction: layerClass is AVCaptureVideoPreviewLayer.
            layer as! AVCaptureVideoPreviewLayer
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer.frame = bounds
        }
    }

    func makeUIView(context: Context) -> UIView {
        let view = PreviewContainerView(frame: .zero)
        view.backgroundColor = .black
        view.previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        guard let previewView = uiView as? PreviewContainerView else { return }
        if previewView.previewLayer.session !== session {
            previewView.previewLayer.session = session
        }
        previewView.previewLayer.frame = previewView.bounds
    }
}
