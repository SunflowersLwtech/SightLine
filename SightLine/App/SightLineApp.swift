//
//  SightLineApp.swift
//  SightLine
//
//  Main app entry point. Uses AppDelegate for audio session setup.
//

import SwiftUI

@main
struct SightLineApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    init() {
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("-uitest-reset-onboarding") {
            UserDefaults.standard.removeObject(forKey: "hasCompletedOnboarding")
        }
        if arguments.contains("-uitest-complete-onboarding") {
            UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
        }
    }

    var body: some Scene {
        WindowGroup {
            if hasCompletedOnboarding {
                MainView()
            } else {
                OnboardingView(hasCompletedOnboarding: $hasCompletedOnboarding)
            }
        }
    }
}
