//
//  ContentView.swift
//  SightLineWatch
//
//  Minimal watchOS UI: large heart rate display + start/stop toggle.
//  Designed for accessibility — all elements have VoiceOver labels.
//  High-contrast colors on dark background for low-vision users.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var workoutManager: WorkoutManager

    var body: some View {
        ScrollView {
            VStack(spacing: 8) {
                // Connection status indicator
                HStack(spacing: 4) {
                    Circle()
                        .fill(PhoneConnector.shared.isReachable ? Color.green : Color.orange)
                        .frame(width: 8, height: 8)

                    Text(PhoneConnector.shared.isReachable ? "iPhone Connected" : "iPhone Unreachable")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(
                    PhoneConnector.shared.isReachable
                        ? "iPhone connected"
                        : "iPhone not reachable"
                )

                // Heart icon
                Image(systemName: workoutManager.isRunning ? "heart.fill" : "heart")
                    .font(.title2)
                    .foregroundColor(.red)
                    .symbolEffect(.pulse, isActive: workoutManager.isRunning)
                    .accessibilityHidden(true)
                    .padding(.top, 8)

                // Heart rate display — large, high-contrast
                Text(heartRateText)
                    .font(.system(size: 52, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                    .contentTransition(.numericText())
                    .accessibilityLabel("Heart rate: \(heartRateAccessibilityText)")

                Text("BPM")
                    .font(.caption)
                    .foregroundColor(.gray)
                    .accessibilityHidden(true)

                // Stability and heading context
                if workoutManager.isRunning {
                    if workoutManager.watchMotion.stabilityScore < 1.0 {
                        Text("Stability: \(Int(workoutManager.watchMotion.stabilityScore * 100))%")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .accessibilityLabel("Stability \(Int(workoutManager.watchMotion.stabilityScore * 100)) percent")
                    }
                    if let heading = workoutManager.watchHeading.heading {
                        Text("Heading: \(Int(heading))\u{00B0}")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .accessibilityLabel("Heading \(Int(heading)) degrees")
                    }
                }

                // Start / Stop button
                Button(action: toggleWorkout) {
                    Label(
                        workoutManager.isRunning ? "Stop" : "Start",
                        systemImage: workoutManager.isRunning ? "stop.fill" : "play.fill"
                    )
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                }
                .tint(workoutManager.isRunning ? .red : .green)
                .padding(.top, 8)
                .accessibilityLabel(
                    workoutManager.isRunning
                        ? "Stop heart rate monitoring"
                        : "Start heart rate monitoring"
                )
                .accessibilityHint(
                    workoutManager.isRunning
                        ? "Double tap to stop the workout session"
                        : "Double tap to start monitoring your heart rate"
                )
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Helpers

    private var heartRateText: String {
        workoutManager.heartRate > 0
            ? "\(Int(workoutManager.heartRate))"
            : "--"
    }

    private var heartRateAccessibilityText: String {
        workoutManager.heartRate > 0
            ? "\(Int(workoutManager.heartRate)) beats per minute"
            : "not available"
    }

    private func toggleWorkout() {
        if workoutManager.isRunning {
            workoutManager.stopWorkout()
        } else {
            workoutManager.startWorkout()
        }
    }
}
