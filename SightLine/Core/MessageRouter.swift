//
//  MessageRouter.swift
//  SightLine
//

import SwiftUI
import os

private let logger = Logger(subsystem: "com.sightline.app", category: "MessageRouter")

@MainActor
final class MessageRouter {
    weak var coordinator: PipelineCoordinator?

    init(coordinator: PipelineCoordinator) {
        self.coordinator = coordinator
    }

    func handleDownstreamMessage(_ msg: DownstreamMessage) {
        guard let coordinator else { return }

        switch msg {
        case .sessionReady:
            logger.info("Session ready, starting audio capture (camera deferred)")
            DispatchQueue.main.async {
                coordinator.hasReceivedSessionReady = true
                coordinator.cancelSessionReadyTimeout()
                // Pre-set model speaking timestamp so silence gate covers
                // the 200-600ms greeting generation delay, preventing
                // ambient noise from reaching Gemini VAD and interrupting the greeting.
                coordinator.audioCapture.lastModelAudioReceivedAt = CFAbsoluteTimeGetCurrent()
                coordinator.startAudioCapture()
            }
        case .faceLibraryReloaded(let count):
            DispatchQueue.main.async {
                let message = "Face library reloaded (\(count) faces)."
                coordinator.transcript = message
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureTranscript(text: message, role: "system")
                }
            }
            logger.info("Face library reloaded: \(count)")
        case .faceLibraryCleared(let deletedCount):
            DispatchQueue.main.async {
                let message = "Face library cleared (\(deletedCount) deleted)."
                coordinator.transcript = message
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureTranscript(text: message, role: "system")
                }
            }
            logger.info("Face library cleared: \(deletedCount)")
        case .error(let message):
            DispatchQueue.main.async {
                coordinator.cancelSessionReadyTimeout()
                coordinator.connectionStatus = "Server error — retrying..."
                coordinator.transcript = "Server error: \(message)"
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureTranscript(text: "Server error: \(message)", role: "system")
                }
                coordinator.audioCapture.stopCapture()
                coordinator.cameraManager.stopCapture()
            }
            logger.error("Server error message: \(message, privacy: .public)")
            coordinator.webSocketManager.reconnect(afterMs: 3000)
        case .transcript(let text, let role):
            DispatchQueue.main.async {
                if role == "agent" {
                    // Agent transcripts → visible as subtitles + dev console
                    coordinator.transcript = text
                    coordinator.lastAgentTranscript = text
                    if coordinator.shouldCaptureDevEvents {
                        coordinator.devConsoleModel.captureTranscript(text: text, role: role)
                    }
                    self.drainWhenIdleToolQueueIfPossible()
                } else {
                    // User / echo transcripts → dev console only (no main UI)
                    if coordinator.shouldCaptureDevEvents {
                        coordinator.devConsoleModel.captureTranscript(text: text, role: role == "user" && coordinator.audioPlayback.isPlaying ? "echo" : role)
                    }
                }
            }
        case .lodUpdate(let lod):
            DispatchQueue.main.async {
                let lodNames = [1: "Safety", 2: "Balanced", 3: "Detailed"]
                if lod != coordinator.currentLOD {
                    UIAccessibility.post(notification: .announcement, argument: "Detail level \(lod): \(lodNames[lod] ?? "")")
                }
                coordinator.currentLOD = lod
                coordinator.frameSelector.updateLOD(lod)
                coordinator.telemetryAggregator.updateLOD(lod)
                coordinator.debugModel.currentLOD = lod
            }
        case .toolEvent(let tool, let behavior, let payload):
            let status = (payload["status"] as? String)?.lowercased() ?? ""
            if !status.isEmpty {
                DispatchQueue.main.async {
                    if coordinator.shouldCaptureDevEvents {
                        coordinator.devConsoleModel.captureTranscript(text: "Tool \(tool): \(status)", role: "system")
                        coordinator.devConsoleModel.captureToolActivity(
                            tool: tool,
                            behavior: behavior.rawValue,
                            status: status
                        )
                    }
                }
            }
            switch status {
            case "queued", "invoked", "completed":
                // Keep these in dev console only; avoid freezing user-facing transcript on tool progress text.
                return
            case "error", "unavailable":
                handleToolMessage(text: "Tool \(tool) is unavailable.", behavior: .WHEN_IDLE)
            default:
                handleToolMessage(
                    text: "Tool update: \(tool)",
                    behavior: behavior
                )
            }
        case .visionResult(let summary, let behavior, let spatialObjects):
            DispatchQueue.main.async { coordinator.debugModel.markCapabilityReady("vision") }
            // Haptic feedback for safety-critical vision alerts
            if behavior == .INTERRUPT {
                HapticManager.shared.obstacleProximity(distance: 0.3)
            }
            // Dispatch haptics based on spatial objects
            self.dispatchSpatialHaptics(spatialObjects)
            handleToolMessage(
                text: summary.isEmpty ? "Vision analysis updated." : summary,
                behavior: behavior
            )
        case .ocrResult(let summary, let behavior):
            DispatchQueue.main.async { coordinator.debugModel.markCapabilityReady("ocr") }
            handleToolMessage(
                text: summary.isEmpty ? "OCR result received." : summary,
                behavior: behavior
            )
        case .visionDebug(let data):
            DispatchQueue.main.async {
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureVisionDebug(data)
                }
            }
        case .ocrDebug(let data):
            DispatchQueue.main.async {
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureOCRDebug(data)
                }
            }
        case .faceDebug(let data):
            DispatchQueue.main.async {
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureFaceDebug(data)
                }
            }
        case .frameAck(let frameId, let queuedAgents):
            DispatchQueue.main.async {
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureFrameAck(frameId: frameId, queuedAgents: queuedAgents)
                }
            }
        case .navigationResult(let summary, let behavior):
            // Haptic feedback for urgent navigation (LOD 1 safety mode)
            if coordinator.currentLOD <= 1 && behavior == .INTERRUPT {
                HapticManager.shared.directionalCue(.ahead)
            }
            handleToolMessage(
                text: summary.isEmpty ? "Navigation result received." : summary,
                behavior: behavior,
                isUrgent: coordinator.currentLOD <= 1
            )
        case .searchResult(let summary, let behavior):
            handleToolMessage(
                text: summary.isEmpty ? "Search result received." : summary,
                behavior: behavior
            )
        case .personIdentified(let name, let behavior):
            DispatchQueue.main.async { coordinator.debugModel.markCapabilityReady("face") }
            handleIdentityMessage(name: name, matched: true, behavior: behavior)
        case .identityUpdate(let name, let matched, let behavior):
            DispatchQueue.main.async { coordinator.debugModel.markCapabilityReady("face") }
            handleIdentityMessage(name: name, matched: matched, behavior: behavior)
        case .capabilityDegraded(let capability, let reason, _):
            DispatchQueue.main.async {
                coordinator.debugModel.markCapabilityDegraded(capability)
                self.handleToolMessage(
                    text: coordinator.capabilityDegradedMessage(for: capability),
                    behavior: .WHEN_IDLE
                )
            }
            logger.warning("Capability degraded: \(capability, privacy: .public) - \(reason, privacy: .public)")
        case .debugLod(let data):
            let memoryTop3 = (data["memory_top3"] as? [String]) ?? []
            DispatchQueue.main.async {
                // SL-77: explicit debugLod -> memory top3 injection for overlay gate.
                coordinator.debugModel.memoryTop3 = Array(memoryTop3.prefix(3))
                coordinator.debugModel.updateFromLodDebug(data)
            }
        case .debugActivity(let data):
            DispatchQueue.main.async {
                coordinator.debugModel.updateFromActivityDebug(data)
            }
        case .interrupted:
            logger.info("Model output interrupted — flushing playback buffer")
            DispatchQueue.main.async {
                coordinator.audioCapture.lastModelAudioReceivedAt = 0  // Expire echo gate immediately
                coordinator.audioPlayback.stopImmediately()
                coordinator.audioPlayback.suppressIncomingAudio(for: 0.8)
                HapticManager.shared.doubleTap()
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureTranscript(
                        text: "Model interrupted by user", role: "system")
                }
            }
        case .goAway(let retryMs):
            logger.info("GoAway received, reconnecting in \(retryMs)ms")
            DispatchQueue.main.async {
                coordinator.cancelSessionReadyTimeout()
                coordinator.connectionStatus = "Reconnecting..."
            }
            coordinator.webSocketManager.reconnect(afterMs: retryMs)
        case .sessionResumption(let handle):
            guard !handle.isEmpty else { return }
            DispatchQueue.main.async {
                coordinator.sessionResumptionHandle = handle
            }
            UserDefaults.standard.set(handle, forKey: SightLineConfig.sessionResumptionHandleDefaultsKey)
            logger.info("Session resumption handle updated: \(handle.prefix(20))...")
        case .thinkingSound(let state):
            DispatchQueue.main.async {
                switch state {
                case "thinking": coordinator.thinkingSoundManager.setState(.thinking)
                case "searching": coordinator.thinkingSoundManager.setState(.searching)
                default: coordinator.thinkingSoundManager.setState(.idle)
                }
            }
        case .toolsManifest(let tools, let modules, let agents):
            DispatchQueue.main.async {
                coordinator.devConsoleModel.captureToolsManifest(tools: tools, modules: modules, subAgents: agents)
            }
        case .profileUpdatedAck:
            logger.info("Profile update acknowledged by server")
        case .unknown(let raw):
            logger.debug("Unknown downstream message: \(String(raw.prefix(200)), privacy: .public)")
            DispatchQueue.main.async {
                if coordinator.shouldCaptureDevEvents {
                    coordinator.devConsoleModel.captureTranscript(
                        text: "Unknown downstream: \(String(raw.prefix(160)))",
                        role: "system"
                    )
                }
            }
        default:
            break
        }
    }

    func handleToolMessage(
        text: String,
        behavior: ToolBehaviorMode,
        isUrgent: Bool = false
    ) {
        guard let coordinator else { return }

        DispatchQueue.main.async {
            let effectiveBehavior: ToolBehaviorMode
            if behavior == .INTERRUPT && coordinator.audioPlayback.isPlaying && !isUrgent {
                // Guardrail: non-urgent tool events should not cut an utterance in half.
                effectiveBehavior = .WHEN_IDLE
                logger.info("Downgraded non-urgent INTERRUPT to WHEN_IDLE while audio is playing")
            } else {
                effectiveBehavior = behavior
            }

            switch effectiveBehavior {
            case .INTERRUPT:
                // Urgent INTERRUPT must stop ongoing playback immediately.
                coordinator.audioPlayback.stopImmediately()
                coordinator.transcript = text
            case .WHEN_IDLE:
                // WHEN_IDLE respects current playback state and queues updates.
                if coordinator.audioPlayback.isPlaying {
                    coordinator.whenIdleToolQueue.append(text)
                } else {
                    coordinator.transcript = text
                }
            case .SILENT:
                logger.debug("SILENT tool update received")
            }
            self.drainWhenIdleToolQueueIfPossible()
        }
    }

    func handleIdentityMessage(name: String, matched: Bool, behavior: ToolBehaviorMode) {
        let personText = matched ? "Person identified: \(name)" : "Identity update available."
        if behavior == .SILENT {
            // identify_person must support SILENT path and avoid hard interruption.
            logger.debug("identity SILENT update for \(name, privacy: .public)")
            return
        }
        handleToolMessage(text: personText, behavior: behavior)
    }

    /// Dispatch haptic feedback based on spatial objects from vision analysis.
    func dispatchSpatialHaptics(_ spatialObjects: [[String: Any]]) {
        for obj in spatialObjects {
            guard let label = obj["label"] as? String,
                  let salience = obj["salience"] as? String else { continue }
            let distance = obj["distance_estimate"] as? String ?? ""
            let isClose = distance == "within_reach" || distance == "1m"

            // Only fire haptics for nearby safety/interaction objects
            guard salience == "safety" || isClose else { continue }

            switch label {
            case "person":
                if isClose { HapticManager.shared.objectTexture(.person) }
            case "vehicle":
                HapticManager.shared.objectTexture(.vehicle)
            case "stairs", "steps":
                HapticManager.shared.objectTexture(.stairs)
            case "door":
                HapticManager.shared.objectTexture(.door)
            default:
                if salience == "safety" {
                    let dist: Float = isClose ? 0.5 : 1.5
                    HapticManager.shared.obstacleProximity(distance: dist)
                }
            }

            // Only dispatch one haptic per vision frame to avoid fatigue
            break
        }
    }

    func drainWhenIdleToolQueueIfPossible() {
        guard let coordinator else { return }
        guard !coordinator.audioPlayback.isPlaying else { return }
        guard !coordinator.whenIdleToolQueue.isEmpty else { return }
        coordinator.transcript = coordinator.whenIdleToolQueue.last ?? coordinator.transcript
        coordinator.whenIdleToolQueue.removeAll()
    }
}
