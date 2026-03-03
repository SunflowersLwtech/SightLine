//
//  DeveloperConsoleView.swift
//  SightLine
//
//  Developer-only debug console for testing and demos.
//  Dark theme, monospace text, tab-based sections.
//  Reads state from existing managers — does NOT create duplicates.
//
//  Entry point: 4-finger tap on MainView, or gear icon in DEBUG builds.
//

import SwiftUI
import Combine
import os

private let logger = Logger(subsystem: "com.sightline.app", category: "DevConsole")

// MARK: - Developer Console Model

@MainActor
final class DeveloperConsoleModel: ObservableObject {
    struct TranscriptEntry: Identifiable {
        let id = UUID()
        let timestamp: Date
        let role: String
        let text: String
    }

    struct NetworkEvent: Identifiable {
        let id = UUID()
        let timestamp: Date
        let direction: String
        let payload: String
    }

    struct DebugBoundingBox: Identifiable {
        let id = UUID()
        let source: String
        let label: String
        let confidence: Double
        let normalizedRect: CGRect
    }

    struct ToolInfo: Identifiable {
        let id = UUID()
        let name: String
        let category: String
        let behavior: String  // INTERRUPT / WHEN_IDLE / SILENT
        let description: String
        var status: String = "ready"  // ready / active / error
    }

    struct ContextModuleInfo: Identifiable {
        let id = UUID()
        let name: String
        var status: String  // ready / unavailable / error
    }

    struct ToolActivityEvent: Identifiable {
        let id = UUID()
        let timestamp: Date
        let tool: String
        let behavior: String
        let status: String  // queued / in_progress / success / error
    }

    @Published var transcripts: [TranscriptEntry] = []
    @Published var networkEvents: [NetworkEvent] = []

    // Sensor data
    @Published var motionState: String = "unknown"
    @Published var heartRate: Double?
    @Published var stepCadence: Double = 0
    @Published var noiseDb: Double = 50.0
    @Published var latitude: Double = 0
    @Published var longitude: Double = 0
    @Published var gpsAccuracy: Double = 0
    @Published var gpsSpeed: Double = 0
    @Published var heading: Double = 0
    @Published var timeContext: String = "unknown"

    // LOD
    @Published var currentLOD: Int = 2
    @Published var lodHistory: [(Date, Int, String)] = []
    @Published var triggeredRules: [String] = []

    // Connection
    @Published var isConnected: Bool = false
    @Published var isSafeMode: Bool = false

    // Sub-agent capabilities
    @Published var visionStatus: String = "ready"
    @Published var ocrStatus: String = "ready"
    @Published var faceStatus: String = "ready"

    // Memory
    @Published var memoryTop3: [String] = []
    @Published var memoryTop3Detailed: [[String: Any]] = []

    // Watch
    @Published var isWatchReachable: Bool = false
    @Published var isWatchMonitoring: Bool = false

    // Camera
    @Published var isCameraRunning: Bool = false
    @Published var visionBoxes: [DebugBoundingBox] = []
    @Published var ocrBoxes: [DebugBoundingBox] = []
    @Published var faceBoxes: [DebugBoundingBox] = []
    @Published var lastFrameAckId: Int = -1
    @Published var lastFrameQueuedAgents: [String] = []

    // Tools manifest
    @Published var registeredTools: [ToolInfo] = []
    @Published var contextModules: [ContextModuleInfo] = []
    @Published var toolActivity: [ToolActivityEvent] = []

    private var cancellables = Set<AnyCancellable>()

    /// Bind to existing managers via Combine. Called once from MainView.onAppear.
    /// Does NOT replace any callbacks — reads only via @Published subscriptions.
    func bind(
        sensorManager: SensorManager,
        cameraManager: CameraManager,
        debugModel: DebugOverlayModel,
        frameSelector: FrameSelector? = nil
    ) {
        // Frame rate -> DebugOverlay
        if let fs = frameSelector {
            fs.$effectiveFPS
                .receive(on: DispatchQueue.main)
                .sink { v in debugModel.frameRate = v }
                .store(in: &cancellables)
        }
        // DebugOverlayModel mirrors (already fed by MainView pipeline)
        debugModel.$currentLOD
            .receive(on: DispatchQueue.main)
            .sink { [weak self] lod in self?.currentLOD = lod }
            .store(in: &cancellables)

        debugModel.$isConnected
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.isConnected = v }
            .store(in: &cancellables)

        debugModel.$isSafeMode
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.isSafeMode = v }
            .store(in: &cancellables)

        debugModel.$visionStatus
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.visionStatus = v }
            .store(in: &cancellables)

        debugModel.$ocrStatus
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.ocrStatus = v }
            .store(in: &cancellables)

        debugModel.$faceStatus
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.faceStatus = v }
            .store(in: &cancellables)

        debugModel.$memoryTop3
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.memoryTop3 = v }
            .store(in: &cancellables)

        debugModel.$memoryTop3Detailed
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.memoryTop3Detailed = v }
            .store(in: &cancellables)

        debugModel.$triggeredRules
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.triggeredRules = v }
            .store(in: &cancellables)

        debugModel.$motionState
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.motionState = v }
            .store(in: &cancellables)

        debugModel.$noiseDb
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.noiseDb = v }
            .store(in: &cancellables)

        debugModel.$stepCadence
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.stepCadence = v }
            .store(in: &cancellables)

        // GPS + heading from LocationManager
        sensorManager.locationManager.$latitude
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in
                self?.latitude = v
                debugModel.latitude = v
            }
            .store(in: &cancellables)

        sensorManager.locationManager.$longitude
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in
                self?.longitude = v
                debugModel.longitude = v
            }
            .store(in: &cancellables)

        sensorManager.locationManager.$accuracy
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.gpsAccuracy = v }
            .store(in: &cancellables)

        sensorManager.locationManager.$speed
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.gpsSpeed = v }
            .store(in: &cancellables)

        sensorManager.locationManager.$heading
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.heading = v }
            .store(in: &cancellables)

        // Watch state
        sensorManager.watchReceiver.$isWatchReachable
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.isWatchReachable = v }
            .store(in: &cancellables)

        sensorManager.watchReceiver.$isWatchMonitoring
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.isWatchMonitoring = v }
            .store(in: &cancellables)

        // Camera running state
        cameraManager.$isRunning
            .receive(on: DispatchQueue.main)
            .sink { [weak self] v in self?.isCameraRunning = v }
            .store(in: &cancellables)

        // Time context
        sensorManager.$currentTelemetry
            .receive(on: DispatchQueue.main)
            .sink { [weak self] data in
                self?.timeContext = data.timeContext
                self?.heartRate = data.heartRate
                self?.motionState = data.motionState
                self?.stepCadence = Double(data.stepCadence)
                self?.noiseDb = data.ambientNoiseDb
            }
            .store(in: &cancellables)

        // LOD history tracking
        debugModel.$currentLOD
            .combineLatest(debugModel.$lodReason)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] lod, reason in
                guard let self = self else { return }
                let last = self.lodHistory.last
                if last == nil || last?.1 != lod {
                    self.lodHistory.append((Date(), lod, reason))
                    if self.lodHistory.count > 50 {
                        self.lodHistory.removeFirst()
                    }
                }
            }
            .store(in: &cancellables)
    }

    /// Called from MainView.handleDownstreamMessage to capture transcripts
    /// without intercepting the WebSocket callback.
    func captureTranscript(text: String, role: String) {
        let entry = TranscriptEntry(timestamp: Date(), role: role, text: text)
        transcripts.append(entry)
        if transcripts.count > 200 {
            transcripts.removeFirst()
        }
    }

    func captureNetworkMessage(direction: String, payload: String) {
        let entry = NetworkEvent(timestamp: Date(), direction: direction, payload: payload)
        networkEvents.append(entry)
        if networkEvents.count > 400 {
            networkEvents.removeFirst()
        }
    }

    func captureVisionDebug(_ data: [String: Any]) {
        visionBoxes = extractBoxes(
            from: data,
            candidateKeys: ["bounding_boxes", "boxes", "objects"],
            source: "vision",
            defaultLabel: "object"
        )
    }

    func captureOCRDebug(_ data: [String: Any]) {
        ocrBoxes = extractBoxes(
            from: data,
            candidateKeys: ["text_regions", "regions", "boxes"],
            source: "ocr",
            defaultLabel: "text"
        )
    }

    func captureFaceDebug(_ data: [String: Any]) {
        faceBoxes = extractBoxes(
            from: data,
            candidateKeys: ["face_boxes", "faces", "detections"],
            source: "face",
            defaultLabel: "face"
        )
    }

    func captureFrameAck(frameId: Int, queuedAgents: [String]) {
        lastFrameAckId = frameId
        lastFrameQueuedAgents = queuedAgents
    }

    func captureToolsManifest(tools: [[String: Any]], modules: [[String: Any]], subAgents: [String: String]) {
        registeredTools = tools.map { t in
            ToolInfo(
                name: t["name"] as? String ?? "unknown",
                category: t["category"] as? String ?? "unknown",
                behavior: t["behavior"] as? String ?? "WHEN_IDLE",
                description: t["description"] as? String ?? ""
            )
        }
        contextModules = modules.map { m in
            ContextModuleInfo(
                name: m["name"] as? String ?? "unknown",
                status: m["status"] as? String ?? "unavailable"
            )
        }
        // Update sub-agent status from manifest
        if let v = subAgents["vision"] { visionStatus = v }
        if let o = subAgents["ocr"] { ocrStatus = o }
        if let f = subAgents["face"] { faceStatus = f }
    }

    func captureToolActivity(tool: String, behavior: String, status: String) {
        let event = ToolActivityEvent(
            timestamp: Date(),
            tool: tool,
            behavior: behavior,
            status: status
        )
        toolActivity.insert(event, at: 0)
        if toolActivity.count > 50 {
            toolActivity.removeLast()
        }
        // Update matching tool's status in registeredTools
        if let idx = registeredTools.firstIndex(where: { $0.name == tool }) {
            var updated = registeredTools[idx]
            switch status {
            case "queued", "invoked", "in_progress":
                updated.status = "active"
            case "error", "unavailable":
                updated.status = "error"
            default:
                updated.status = "ready"
            }
            registeredTools[idx] = updated
        }
    }

    func clearNetworkEvents() {
        networkEvents.removeAll()
    }

    private func extractBoxes(
        from data: [String: Any],
        candidateKeys: [String],
        source: String,
        defaultLabel: String
    ) -> [DebugBoundingBox] {
        let candidates = candidatePayloadArray(from: data, keys: candidateKeys)
        return candidates.compactMap { item in
            guard let rect = parseNormalizedRect(from: item) else { return nil }
            let label = (item["label"] as? String)
                ?? (item["name"] as? String)
                ?? (item["person_name"] as? String)
                ?? defaultLabel
            let confidence = (item["confidence"] as? Double)
                ?? (item["score"] as? Double)
                ?? (item["similarity"] as? Double)
                ?? 0.0
            return DebugBoundingBox(
                source: source,
                label: label,
                confidence: confidence,
                normalizedRect: rect
            )
        }
    }

    private func candidatePayloadArray(from data: [String: Any], keys: [String]) -> [[String: Any]] {
        for key in keys {
            if let items = data[key] as? [[String: Any]] {
                return items
            }
        }
        if let nested = data["data"] as? [String: Any] {
            for key in keys {
                if let items = nested[key] as? [[String: Any]] {
                    return items
                }
            }
        }
        return []
    }

    private func parseNormalizedRect(from item: [String: Any]) -> CGRect? {
        if let box2D = item["box_2d"] as? [Double], box2D.count == 4 {
            let scale = (box2D.max() ?? 1.0) > 1.0 ? 1000.0 : 1.0
            return normalizedRect(
                xmin: box2D[1] / scale,
                ymin: box2D[0] / scale,
                xmax: box2D[3] / scale,
                ymax: box2D[2] / scale
            )
        }

        if let bbox = item["bbox"] as? [Double], bbox.count == 4 {
            let maxVal = bbox.max() ?? 1.0
            let scale = maxVal > 1.0 ? 768.0 : 1.0
            return normalizedRect(
                xmin: bbox[0] / scale,
                ymin: bbox[1] / scale,
                xmax: bbox[2] / scale,
                ymax: bbox[3] / scale
            )
        }

        return nil
    }

    private func normalizedRect(xmin: Double, ymin: Double, xmax: Double, ymax: Double) -> CGRect? {
        let clampedMinX = min(max(xmin, 0), 1)
        let clampedMinY = min(max(ymin, 0), 1)
        let clampedMaxX = min(max(xmax, 0), 1)
        let clampedMaxY = min(max(ymax, 0), 1)
        let width = clampedMaxX - clampedMinX
        let height = clampedMaxY - clampedMinY
        guard width > 0, height > 0 else { return nil }
        return CGRect(x: clampedMinX, y: clampedMinY, width: width, height: height)
    }
}

// MARK: - Developer Console View

struct DeveloperConsoleView: View {
    @ObservedObject var model: DeveloperConsoleModel
    @ObservedObject var webSocketManager: WebSocketManager
    @ObservedObject var cameraManager: CameraManager
    @Binding var isMuted: Bool
    @Binding var isPaused: Bool

    @State private var selectedTab = 0
    @Environment(\.dismiss) private var dismiss

    private let tabs = ["Live", "Context", "Sensors", "Controls", "Network"]

    var body: some View {
        VStack(spacing: 0) {
            header
            tabBar

            TabView(selection: $selectedTab) {
                liveTab.tag(0)
                contextTab.tag(1)
                sensorsTab.tag(2)
                controlsTab.tag(3)
                networkTab.tag(4)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("DEV CONSOLE")
                .font(.system(size: 14, weight: .bold, design: .monospaced))
                .foregroundColor(.green)

            Spacer()

            connectionBadge

            Spacer()

            Button(action: { dismiss() }) {
                Text("CLOSE")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundColor(.red)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.black)
    }

    private var connectionBadge: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(model.isSafeMode ? .red : (model.isConnected ? .green : .yellow))
                .frame(width: 6, height: 6)
            Text(model.isSafeMode ? "SAFE" : (model.isConnected ? "CONN" : "DISC"))
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
            Text("LOD \(model.currentLOD)")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(lodColor)
        }
    }

    // MARK: - Tab Bar

    private var tabBar: some View {
        HStack(spacing: 0) {
            ForEach(Array(tabs.enumerated()), id: \.offset) { index, title in
                Button(action: { withAnimation(.easeInOut(duration: 0.15)) { selectedTab = index } }) {
                    VStack(spacing: 4) {
                        Text(title)
                            .font(.system(size: 12, weight: selectedTab == index ? .bold : .regular, design: .monospaced))
                            .foregroundColor(selectedTab == index ? .green : .white.opacity(0.5))
                        Rectangle()
                            .fill(selectedTab == index ? Color.green : Color.clear)
                            .frame(height: 2)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 8)
                }
            }
        }
        .background(Color.white.opacity(0.04))
    }

    // MARK: - Tab 0: Live (Conversation Log)

    private var liveTab: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(model.transcripts) { entry in
                        HStack(alignment: .top, spacing: 6) {
                            Text(formatTime(entry.timestamp))
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(.white.opacity(0.3))

                            Text(entry.role.uppercased())
                                .font(.system(size: 9, weight: .bold, design: .monospaced))
                                .foregroundColor(entry.role == "echo" ? .gray : (entry.role == "user" ? .cyan : (entry.role == "gesture" ? .yellow : .green)))
                                .frame(width: 40, alignment: .leading)

                            Text(entry.text)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.white.opacity(0.85))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .id(entry.id)
                    }
                }
                .padding(8)
            }
            .onChange(of: model.transcripts.count) { _, _ in
                if let last = model.transcripts.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    // MARK: - Tab 1: Context (Tools + Pipeline + Activity)

    private var contextTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Section A: Tool Registry
                toolRegistrySection

                // Section B: Context Pipeline
                contextPipelineSection

                // Section C: Tool Activity Stream
                toolActivitySection
            }
            .padding(10)
        }
    }

    // MARK: Tool Registry

    private var toolRegistrySection: some View {
        let grouped = Dictionary(grouping: model.registeredTools, by: { $0.category })
        let readyCount = model.registeredTools.filter { $0.status == "ready" || $0.status == "active" }.count
        let totalCount = model.registeredTools.count

        return sectionBox {
            VStack(alignment: .leading, spacing: 8) {
                iconSectionHeader("TOOLS \(readyCount)/\(totalCount)", icon: "wrench.and.screwdriver")

                if model.registeredTools.isEmpty {
                    Text("Waiting for manifest...")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.3))
                } else {
                    ForEach(grouped.keys.sorted(), id: \.self) { category in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(category.uppercased())
                                .font(.system(size: 9, weight: .bold, design: .monospaced))
                                .foregroundColor(.white.opacity(0.35))
                                .tracking(0.5)

                            ForEach(grouped[category] ?? []) { tool in
                                toolRow(tool)
                            }
                        }
                        .padding(.bottom, 4)
                    }
                }
            }
        }
    }

    private func toolRow(_ tool: DeveloperConsoleModel.ToolInfo) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(toolStatusColor(tool.status))
                .frame(width: 6, height: 6)
            Text(tool.name)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))
                .lineLimit(1)
            Spacer()
            behaviorBadge(tool.behavior)
        }
    }

    // MARK: Context Pipeline

    private var contextPipelineSection: some View {
        sectionBox {
            VStack(alignment: .leading, spacing: 8) {
                iconSectionHeader("CONTEXT PIPELINE", icon: "cpu")

                // Context modules grid
                if model.contextModules.isEmpty {
                    Text("No context modules loaded")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.3))
                } else {
                    LazyVGrid(columns: [
                        GridItem(.flexible(), spacing: 8),
                        GridItem(.flexible(), spacing: 8)
                    ], spacing: 6) {
                        ForEach(model.contextModules) { module in
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(module.status == "ready" ? Color.green : Color.gray)
                                    .frame(width: 6, height: 6)
                                Text(module.name)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.7))
                                    .lineLimit(1)
                                Spacer()
                            }
                        }
                    }
                }

                Divider().background(Color.white.opacity(0.08))

                // LOD + Sub-agents inline
                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        Text("LOD")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(.white.opacity(0.5))
                        Text("\(model.currentLOD)")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(lodColor)
                    }

                    Spacer()

                    HStack(spacing: 6) {
                        subAgentIndicator("V", status: model.visionStatus)
                        subAgentIndicator("O", status: model.ocrStatus)
                        subAgentIndicator("F", status: model.faceStatus)
                    }
                }

                Divider().background(Color.white.opacity(0.08))

                // Memory top 3
                iconSectionHeader("MEMORY", icon: "brain")
                if model.memoryTop3Detailed.isEmpty && model.memoryTop3.isEmpty {
                    Text("No memories loaded")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.3))
                } else if !model.memoryTop3Detailed.isEmpty {
                    ForEach(Array(model.memoryTop3Detailed.prefix(3).enumerated()), id: \.offset) { idx, memory in
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 4) {
                                Text("\(idx + 1).")
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(.green)
                                Text(memory["category"] as? String ?? "general")
                                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                                    .foregroundColor(.cyan)
                                    .padding(.horizontal, 4)
                                    .background(Color.cyan.opacity(0.15))
                                    .cornerRadius(3)
                                Text(String(format: "imp=%.2f", memory["importance"] as? Double ?? 0.5))
                                    .font(.system(size: 8, design: .monospaced))
                                    .foregroundColor(.yellow)
                                Text(String(format: "score=%.3f", memory["score"] as? Double ?? 0))
                                    .font(.system(size: 8, design: .monospaced))
                                    .foregroundColor(.orange)
                            }
                            Text(memory["content"] as? String ?? "")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.white.opacity(0.7))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                } else {
                    ForEach(Array(model.memoryTop3.enumerated()), id: \.offset) { idx, memory in
                        HStack(alignment: .top, spacing: 4) {
                            Text("\(idx + 1).")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundColor(.green)
                            Text(memory)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.white.opacity(0.7))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                Divider().background(Color.white.opacity(0.08))

                // LOD History
                iconSectionHeader("LOD HISTORY", icon: "chart.line.uptrend.xyaxis")
                ForEach(Array(model.lodHistory.suffix(10).reversed().enumerated()), id: \.offset) { _, entry in
                    HStack(spacing: 6) {
                        Text(formatTime(entry.0))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.white.opacity(0.3))
                        Text("LOD \(entry.1)")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(lodColorFor(entry.1))
                        Text(entry.2)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.white.opacity(0.5))
                            .lineLimit(1)
                    }
                }

                // Triggered Rules
                if !model.triggeredRules.isEmpty {
                    Divider().background(Color.white.opacity(0.08))
                    iconSectionHeader("TRIGGERED RULES", icon: "bolt.fill")
                    ForEach(model.triggeredRules, id: \.self) { rule in
                        Text("- \(rule)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.yellow)
                    }
                }
            }
        }
    }

    // MARK: Tool Activity Stream

    private var toolActivitySection: some View {
        sectionBox {
            VStack(alignment: .leading, spacing: 8) {
                iconSectionHeader("TOOL ACTIVITY", icon: "list.bullet.rectangle")

                if model.toolActivity.isEmpty {
                    Text("No tool events yet")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.3))
                } else {
                    ForEach(model.toolActivity.prefix(30)) { event in
                        HStack(spacing: 6) {
                            Text(formatTime(event.timestamp))
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(.white.opacity(0.3))
                            Text(event.tool)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.white.opacity(0.8))
                                .lineLimit(1)
                            Spacer()
                            behaviorBadge(event.behavior)
                            activityStatusIcon(event.status)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Tab 2: Sensors

    private var sensorsTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Gesture states
                sectionBox {
                    VStack(alignment: .leading, spacing: 6) {
                        iconSectionHeader("GESTURE STATES", icon: "hand.tap")
                        dataRow("Microphone", value: isMuted ? "MUTED" : "ACTIVE",
                                 color: isMuted ? .red : .green)
                        dataRow("Session", value: isPaused ? "PAUSED" : "ACTIVE",
                                 color: isPaused ? .orange : .green)
                        dataRow("Camera", value: model.isCameraRunning ? "ON" : "OFF",
                                 color: model.isCameraRunning ? .green : .red)
                        dataRow("LOD", value: "\(model.currentLOD)", color: lodColor)
                    }
                }

                // Connection
                sectionBox {
                    VStack(alignment: .leading, spacing: 6) {
                        iconSectionHeader("CONNECTION", icon: "antenna.radiowaves.left.and.right")
                        dataRow("WebSocket", value: model.isConnected ? "Connected" : "Disconnected",
                                 color: model.isConnected ? .green : .red)
                        dataRow("Safe Mode", value: model.isSafeMode ? "ACTIVE" : "OFF",
                                 color: model.isSafeMode ? .red : .green)
                    }
                }

                // Camera debug
                sectionBox {
                    VStack(alignment: .leading, spacing: 6) {
                        iconSectionHeader("CAMERA DEBUG", icon: "camera.viewfinder")
                        dataRow("Vision Boxes", value: "\(model.visionBoxes.count)", color: .green)
                        dataRow("OCR Boxes", value: "\(model.ocrBoxes.count)", color: .yellow)
                        dataRow("Face Boxes", value: "\(model.faceBoxes.count)", color: .cyan)
                        dataRow("Last Frame Ack", value: model.lastFrameAckId >= 0 ? "#\(model.lastFrameAckId)" : "--")
                        dataRow(
                            "Queued Agents",
                            value: model.lastFrameQueuedAgents.isEmpty ? "--" : model.lastFrameQueuedAgents.joined(separator: ", ")
                        )
                    }
                }

                // Watch data
                sectionBox {
                    VStack(alignment: .leading, spacing: 6) {
                        iconSectionHeader("WATCH DATA", icon: "applewatch.watchface")
                        dataRow("Heart Rate", value: model.heartRate.map { String(format: "%.0f BPM", $0) } ?? "--")
                        dataRow("Watch Reachable", value: model.isWatchReachable ? "YES" : "NO",
                                 color: model.isWatchReachable ? .green : .red)
                        dataRow("Watch Monitoring", value: model.isWatchMonitoring ? "ACTIVE" : "OFF",
                                 color: model.isWatchMonitoring ? .green : .yellow)
                        dataRow("Motion State", value: model.motionState)
                        dataRow("Step Cadence", value: String(format: "%.0f", model.stepCadence))
                    }
                }

                // GPS
                sectionBox {
                    VStack(alignment: .leading, spacing: 6) {
                        iconSectionHeader("GPS", icon: "location.fill")
                        dataRow("Lat", value: String(format: "%.6f", model.latitude))
                        dataRow("Lng", value: String(format: "%.6f", model.longitude))
                        dataRow("Accuracy", value: String(format: "%.1f m", model.gpsAccuracy))
                        dataRow("Speed", value: String(format: "%.1f m/s", model.gpsSpeed))
                    }
                }

                // Environment
                sectionBox {
                    VStack(alignment: .leading, spacing: 6) {
                        iconSectionHeader("ENVIRONMENT", icon: "cloud.sun")
                        dataRow("Noise", value: String(format: "%.0f dB", model.noiseDb))
                        dataRow("Heading", value: String(format: "%.0f\u{00B0}", model.heading))
                        dataRow("Time Context", value: model.timeContext)
                    }
                }
            }
            .padding(10)
        }
    }

    // MARK: - Tab 3: Debug Controls

    private var controlsTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                sectionBox {
                    VStack(alignment: .leading, spacing: 8) {
                        iconSectionHeader("FORCE LOD", icon: "dial.low")
                        HStack(spacing: 8) {
                            lodButton(1)
                            lodButton(2)
                            lodButton(3)
                        }
                    }
                }

                sectionBox {
                    VStack(alignment: .leading, spacing: 8) {
                        iconSectionHeader("FACE LIBRARY", icon: "person.crop.circle.badge.checkmark")
                        controlButton("Reload Face Library", color: .blue) {
                            webSocketManager.sendText("{\"type\":\"reload_face_library\"}")
                        }
                        controlButton("Clear Face Library", color: .red) {
                            webSocketManager.sendText("{\"type\":\"clear_face_library\"}")
                        }
                    }
                }

                sectionBox {
                    VStack(alignment: .leading, spacing: 8) {
                        iconSectionHeader("WEBSOCKET", icon: "network")
                        controlButton("Reconnect", color: .orange) {
                            let resumeHandle = UserDefaults.standard.string(
                                forKey: SightLineConfig.sessionResumptionHandleDefaultsKey
                            )
                            guard let url = SightLineConfig.wsURL(
                                userId: SightLineConfig.defaultUserId,
                                sessionId: SightLineConfig.defaultSessionId,
                                resumeHandle: resumeHandle
                            ) else { return }
                            webSocketManager.disconnect()
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                webSocketManager.connect(url: url)
                            }
                        }
                    }
                }

                sectionBox {
                    VStack(alignment: .leading, spacing: 8) {
                        iconSectionHeader("DATA", icon: "doc.text")
                        controlButton("Copy Session Info", color: .cyan) {
                            let info = """
                            Session: \(SightLineConfig.defaultSessionId)
                            User: \(SightLineConfig.defaultUserId)
                            Server: \(SightLineConfig.serverBaseURL)
                            LOD: \(model.currentLOD)
                            Connected: \(model.isConnected)
                            Safe Mode: \(model.isSafeMode)
                            """
                            UIPasteboard.general.string = info
                        }
                        controlButton("Clear Transcript Log", color: .yellow) {
                            model.transcripts.removeAll()
                        }
                        controlButton("Clear Network Log", color: .yellow) {
                            model.clearNetworkEvents()
                        }
                    }
                }
            }
            .padding(10)
        }
    }

    // MARK: - Tab 4: Network Debug

    private var networkTab: some View {
        VStack(spacing: 8) {
            HStack {
                iconSectionHeader("WEBSOCKET JSON FLOW", icon: "arrow.left.arrow.right")
                Spacer()
                controlButton("Clear", color: .yellow) {
                    model.clearNetworkEvents()
                }
                .frame(width: 96)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        ForEach(model.networkEvents) { event in
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(spacing: 6) {
                                    Text(formatTime(event.timestamp))
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundColor(.white.opacity(0.35))
                                    Text(event.direction)
                                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                                        .foregroundColor(event.direction == "UP" ? .cyan : .green)
                                }
                                Text(event.payload)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.78))
                                    .textSelection(.enabled)
                            }
                            .id(event.id)
                            .padding(.vertical, 4)
                            .padding(.horizontal, 6)
                            .background(Color.white.opacity(0.04))
                            .clipShape(.rect(cornerRadius: 6))
                        }
                    }
                    .padding(.horizontal, 2)
                    .padding(.bottom, 8)
                }
                .onChange(of: model.networkEvents.count) { _, _ in
                    if let last = model.networkEvents.last {
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }
        }
        .padding(8)
    }

    // MARK: - Shared Subviews

    private func sectionBox<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .padding(10)
            .background(Color.white.opacity(0.03))
            .clipShape(.rect(cornerRadius: 8))
    }

    private func iconSectionHeader(_ title: String, icon: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10))
                .foregroundColor(.white.opacity(0.35))
            Text(title)
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(.white.opacity(0.4))
                .tracking(1)
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .foregroundColor(.white.opacity(0.4))
            .tracking(1)
    }

    private func dataRow(_ label: String, value: String, color: Color = .green) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.6))
            Spacer()
            Text(value)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(color)
        }
    }

    private func capabilityRow(_ name: String, status: String) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(status == "ready" ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(name)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
            Spacer()
            Text(status.uppercased())
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(status == "ready" ? .green : .red)
        }
    }

    private func behaviorBadge(_ behavior: String) -> some View {
        Text(behavior)
            .font(.system(size: 8, weight: .bold, design: .monospaced))
            .foregroundColor(.white)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(behaviorColor(behavior).opacity(0.6))
            .clipShape(Capsule())
    }

    private func subAgentIndicator(_ letter: String, status: String) -> some View {
        HStack(spacing: 2) {
            Text(letter)
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(.white.opacity(0.6))
            Text(status == "ready" ? "\u{2713}" : "\u{2717}")
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(status == "ready" ? .green : .red)
        }
    }

    private func activityStatusIcon(_ status: String) -> some View {
        Text(activityStatusSymbol(status))
            .font(.system(size: 10, design: .monospaced))
            .foregroundColor(activityStatusColor(status))
    }

    private func lodButton(_ lod: Int) -> some View {
        Button(action: {
            webSocketManager.sendText(UpstreamMessage.gesture(type: "force_lod_\(lod)").toJSON())
        }) {
            Text("LOD \(lod)")
                .font(.system(size: 12, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(lodColorFor(lod).opacity(model.currentLOD == lod ? 1.0 : 0.4))
                .cornerRadius(6)
        }
    }

    private func controlButton(_ title: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(color.opacity(0.3))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(color.opacity(0.6), lineWidth: 1)
                )
                .cornerRadius(6)
        }
    }

    // MARK: - Helpers

    private var lodColor: Color { lodColorFor(model.currentLOD) }

    private func lodColorFor(_ lod: Int) -> Color {
        switch lod {
        case 1: return .red
        case 2: return .orange
        case 3: return .green
        default: return .gray
        }
    }

    private func behaviorColor(_ behavior: String) -> Color {
        switch behavior.uppercased() {
        case "INTERRUPT": return .red
        case "WHEN_IDLE": return .blue
        case "SILENT": return .gray
        default: return .gray
        }
    }

    private func toolStatusColor(_ status: String) -> Color {
        switch status {
        case "active": return .yellow
        case "error": return .red
        default: return .green
        }
    }

    private func activityStatusSymbol(_ status: String) -> String {
        switch status {
        case "queued", "invoked": return "\u{23F3}"  // hourglass
        case "completed", "success": return "\u{2713}"  // checkmark
        case "error", "unavailable": return "\u{2717}"  // cross
        default: return "\u{2022}"  // bullet
        }
    }

    private func activityStatusColor(_ status: String) -> Color {
        switch status {
        case "queued", "invoked": return .yellow
        case "completed", "success": return .green
        case "error", "unavailable": return .red
        default: return .white.opacity(0.5)
        }
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: date)
    }
}
