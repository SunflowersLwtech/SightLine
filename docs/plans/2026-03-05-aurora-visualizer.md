# Aurora Visualizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the breathing circle indicator with a Gemini Live-style aurora light effect that visualizes user voice energy and model speaking state.

**Architecture:** A standalone `AuroraVisualizer` SwiftUI view using `TimelineView` + `Canvas` for GPU-accelerated rendering. 3-4 animated gradient blobs composited with additive blending. Two inputs: normalized user mic energy (0-1) and model speaking boolean. Minimal changes to MainView — add one `@State` for energy, wire existing callbacks, swap the view.

**Tech Stack:** SwiftUI (Canvas, TimelineView), CoreGraphics gradients, existing AudioCaptureManager RMS + AudioPlaybackManager.isPlaying

---

### Task 1: Create AuroraVisualizer View

**Files:**
- Create: `SightLine/UI/AuroraVisualizer.swift`

**Step 1: Create the aurora visualizer view**

```swift
// AuroraVisualizer.swift
// Gemini Live-style aurora light effect driven by audio energy.

import SwiftUI

struct AuroraVisualizer: View {
    /// Normalized user microphone energy (0...1)
    var userEnergy: CGFloat
    /// Whether the AI model is currently speaking
    var modelSpeaking: Bool

    // Smoothed values for fluid animation
    @State private var smoothedEnergy: CGFloat = 0
    @State private var displayLink: CADisplayLink?

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let time = timeline.date.timeIntervalSinceReferenceDate
                drawAurora(context: &context, size: size, time: time)
            }
        }
        .drawingGroup()
        .onChange(of: userEnergy) { newVal in
            // Smooth energy changes to prevent jitter
            withAnimation(.linear(duration: 0.08)) {
                smoothedEnergy = newVal
            }
        }
    }

    private func drawAurora(context: inout GraphicsContext, size: CGSize, time: Double) {
        let energy = smoothedEnergy
        let speaking = modelSpeaking

        // Base parameters
        let centerY = size.height * 0.65
        let baseWidth = size.width * 0.7
        let baseHeight = size.height * 0.25

        // Energy multiplier: idle = subtle breathing, active = expansive
        let breathe = CGFloat(0.03 * sin(time * 1.5))
        let energyScale = 0.6 + energy * 0.8 + breathe

        // Color selection based on state
        let colors: [Color]
        let opacity: Double
        if speaking {
            colors = [
                Color(red: 0.1, green: 0.6, blue: 0.9),  // cyan
                Color(red: 0.3, green: 0.4, blue: 0.95),  // blue-purple
                Color(red: 0.15, green: 0.7, blue: 0.85), // teal
            ]
            opacity = 0.7 + 0.3 * (0.5 + 0.5 * sin(time * 3.0))
        } else if energy > 0.05 {
            colors = [
                Color(red: 0.3, green: 0.6, blue: 1.0),   // bright blue
                Color(red: 0.5, green: 0.8, blue: 1.0),   // light blue
                Color(red: 0.9, green: 0.95, blue: 1.0),  // near white
            ]
            opacity = 0.5 + Double(energy) * 0.5
        } else {
            colors = [
                Color(red: 0.15, green: 0.25, blue: 0.5), // dark blue
                Color(red: 0.1, green: 0.2, blue: 0.4),   // darker blue
                Color(red: 0.2, green: 0.3, blue: 0.5),   // muted blue
            ]
            opacity = 0.35 + 0.1 * (0.5 + 0.5 * sin(time * 0.8))
        }

        // Draw 3 blobs with different phase offsets
        for i in 0..<3 {
            let phase = Double(i) * 2.1
            let xOffset = CGFloat(sin(time * 0.7 + phase)) * size.width * 0.15
            let yOffset = CGFloat(cos(time * 0.5 + phase)) * size.height * 0.04
            let scaleVariation = CGFloat(1.0 + 0.15 * sin(time * 1.2 + phase))

            let blobWidth = baseWidth * energyScale * scaleVariation * (speaking ? 1.2 : 1.0)
            let blobHeight = baseHeight * energyScale * scaleVariation

            let rect = CGRect(
                x: size.width / 2 - blobWidth / 2 + xOffset,
                y: centerY - blobHeight / 2 + yOffset,
                width: blobWidth,
                height: blobHeight
            )

            let gradient = Gradient(colors: [
                colors[i % colors.count].opacity(opacity),
                colors[i % colors.count].opacity(opacity * 0.5),
                colors[i % colors.count].opacity(0),
            ])

            context.fill(
                Path(ellipseIn: rect),
                with: .radialGradient(
                    gradient,
                    center: CGPoint(x: rect.midX, y: rect.midY),
                    startRadius: 0,
                    endRadius: max(blobWidth, blobHeight) / 2
                )
            )
        }

        // Glow highlight layer — brighter center blob when energy is high
        if energy > 0.15 || speaking {
            let glowIntensity = speaking ? 0.4 : Double(energy) * 0.6
            let glowWidth = baseWidth * 0.5 * energyScale
            let glowHeight = baseHeight * 0.4 * energyScale
            let glowRect = CGRect(
                x: size.width / 2 - glowWidth / 2,
                y: centerY - glowHeight / 2,
                width: glowWidth,
                height: glowHeight
            )
            let glowGradient = Gradient(colors: [
                Color.white.opacity(glowIntensity),
                Color.white.opacity(glowIntensity * 0.3),
                Color.white.opacity(0),
            ])
            context.fill(
                Path(ellipseIn: glowRect),
                with: .radialGradient(
                    glowGradient,
                    center: CGPoint(x: glowRect.midX, y: glowRect.midY),
                    startRadius: 0,
                    endRadius: max(glowWidth, glowHeight) / 2
                )
            )
        }
    }
}
```

**Step 2: Verify it compiles**

Run: `cd /Users/sunfl/Documents/study/glac/SightLine && xcodebuild -scheme SightLine -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -20`

**Step 3: Commit**

```bash
git add SightLine/UI/AuroraVisualizer.swift
git commit -m "feat(ui): add AuroraVisualizer view with Canvas+TimelineView rendering"
```

---

### Task 2: Wire AuroraVisualizer into MainView

**Files:**
- Modify: `SightLine/UI/MainView.swift`

**Step 1: Add userAudioEnergy state**

Add `@State private var userAudioEnergy: CGFloat = 0` to MainView's state properties.

**Step 2: Replace breathingIndicator usage with AuroraVisualizer**

In `mainContent`, replace the breathing indicator section with the aurora placed at the bottom of the ZStack (behind all other content).

**Step 3: Wire onAudioLevelUpdate to update userAudioEnergy**

In the existing `audioCapture.onAudioLevelUpdate` callback (line ~846), add energy normalization and MainActor update.

**Step 4: Remove the old breathingIndicator computed property**

Delete the `breathingIndicator` property (lines 277-299).

**Step 5: Verify it compiles**

**Step 6: Commit**

```bash
git add SightLine/UI/MainView.swift
git commit -m "feat(ui): replace breathing circle with aurora visualizer"
```
