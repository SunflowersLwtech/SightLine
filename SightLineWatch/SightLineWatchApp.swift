//
//  SightLineWatchApp.swift
//  SightLineWatch
//
//  watchOS Companion App for SightLine.
//  Captures heart rate, motion, heading, and health context via
//  HKWorkoutSession and transmits to iPhone via WCSession (<1s latency)
//  to enrich AI responses with real-time Watch context.
//

import SwiftUI

@main
struct SightLineWatchApp: App {
    @StateObject private var workoutManager = WorkoutManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(workoutManager)
                .onAppear {
                    PhoneConnector.shared.activate()
                }
        }
    }
}
