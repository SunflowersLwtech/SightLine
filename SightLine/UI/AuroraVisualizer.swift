//
//  AuroraVisualizer.swift
//  SightLine
//
//  Gemini Live-style aurora light effect that visualizes audio state.
//  Uses Canvas + TimelineView for GPU-accelerated rendering.
//  Two inputs: normalized user mic energy and model speaking state.
//

import SwiftUI

struct AuroraVisualizer: View {
    /// Normalized user microphone energy (0...1)
    var userEnergy: CGFloat
    /// Whether the AI model is currently speaking
    var modelSpeaking: Bool

    @State private var smoothedEnergy: CGFloat = 0

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let time = timeline.date.timeIntervalSinceReferenceDate
                drawAurora(context: &context, size: size, time: time)
            }
        }
        .drawingGroup()
        .allowsHitTesting(false)
        .accessibilityHidden(true)
        .onChange(of: userEnergy) { _, newVal in
            smoothedEnergy += (newVal - smoothedEnergy) * 0.3
        }
        .onChange(of: modelSpeaking) { _, speaking in
            if !speaking {
                smoothedEnergy *= 0.5
            }
        }
    }

    private func drawAurora(context: inout GraphicsContext, size: CGSize, time: Double) {
        let energy = smoothedEnergy
        let speaking = modelSpeaking

        // Aurora sits in the bottom third
        let centerY = size.height * 0.7
        let baseWidth = size.width * 0.8
        let baseHeight = size.height * 0.3

        // Idle breathing + energy-driven scale
        let breathe = CGFloat(0.03 * sin(time * 1.5))
        let energyScale: CGFloat = 0.5 + energy * 0.9 + breathe

        // Color palette per state
        let colors: [Color]
        let baseOpacity: Double

        if speaking {
            // Model speaking: cyan/blue-purple, pulsing
            colors = [
                Color(red: 0.1, green: 0.6, blue: 0.9),
                Color(red: 0.3, green: 0.4, blue: 0.95),
                Color(red: 0.15, green: 0.7, blue: 0.85),
            ]
            baseOpacity = 0.6 + 0.25 * (0.5 + 0.5 * sin(time * 3.0))
        } else if energy > 0.05 {
            // User speaking: bright blue -> white
            colors = [
                Color(red: 0.3, green: 0.6, blue: 1.0),
                Color(red: 0.5, green: 0.8, blue: 1.0),
                Color(red: 0.85, green: 0.92, blue: 1.0),
            ]
            baseOpacity = 0.4 + Double(energy) * 0.6
        } else {
            // Idle: dark muted blue, gentle breathing
            colors = [
                Color(red: 0.12, green: 0.22, blue: 0.45),
                Color(red: 0.08, green: 0.18, blue: 0.38),
                Color(red: 0.18, green: 0.28, blue: 0.48),
            ]
            baseOpacity = 0.3 + 0.08 * (0.5 + 0.5 * sin(time * 0.8))
        }

        // 3 blobs with staggered phase
        for i in 0..<3 {
            let phase = Double(i) * 2.1
            let xOffset = CGFloat(sin(time * 0.7 + phase)) * size.width * 0.18
            let yOffset = CGFloat(cos(time * 0.5 + phase)) * size.height * 0.04
            let pulse = CGFloat(1.0 + 0.15 * sin(time * 1.2 + phase))

            let blobW = baseWidth * energyScale * pulse * (speaking ? 1.15 : 1.0)
            let blobH = baseHeight * energyScale * pulse

            let rect = CGRect(
                x: size.width / 2 - blobW / 2 + xOffset,
                y: centerY - blobH / 2 + yOffset,
                width: blobW,
                height: blobH
            )

            let color = colors[i % colors.count]
            let gradient = Gradient(colors: [
                color.opacity(baseOpacity),
                color.opacity(baseOpacity * 0.4),
                color.opacity(0),
            ])

            context.fill(
                Path(ellipseIn: rect),
                with: .radialGradient(
                    gradient,
                    center: CGPoint(x: rect.midX, y: rect.midY),
                    startRadius: 0,
                    endRadius: max(blobW, blobH) / 2
                )
            )
        }

        // Central glow highlight when active
        if energy > 0.12 || speaking {
            let glowStrength = speaking
                ? 0.35 + 0.15 * sin(time * 2.5)
                : Double(energy) * 0.55
            let glowW = baseWidth * 0.45 * energyScale
            let glowH = baseHeight * 0.35 * energyScale
            let glowRect = CGRect(
                x: size.width / 2 - glowW / 2,
                y: centerY - glowH / 2,
                width: glowW,
                height: glowH
            )
            context.fill(
                Path(ellipseIn: glowRect),
                with: .radialGradient(
                    Gradient(colors: [
                        Color.white.opacity(glowStrength),
                        Color.white.opacity(glowStrength * 0.25),
                        Color.white.opacity(0),
                    ]),
                    center: CGPoint(x: glowRect.midX, y: glowRect.midY),
                    startRadius: 0,
                    endRadius: max(glowW, glowH) / 2
                )
            )
        }
    }
}
