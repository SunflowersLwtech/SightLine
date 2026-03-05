//
//  OnboardingView.swift
//  SightLine
//
//  Minimal onboarding gate. Shows app intro and a "Get Started" button.
//  Once tapped, sets hasCompletedOnboarding and transitions to MainView.
//  Future expansion: add tutorial steps, permissions walkthrough, etc.
//

import SwiftUI

struct OnboardingView: View {
    private let actionButtonColor = Color(red: 0.0, green: 0.20, blue: 0.56)

    @Binding var hasCompletedOnboarding: Bool
    @State private var showProfileSetup = false
    @State private var showControlsGuide = false

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Image(systemName: "eye.slash.circle.fill")
                    .font(.system(size: 80))
                    .foregroundStyle(.blue)
                    .accessibilityHidden(true)
                    .padding(.top, 24)

                Text("SightLine")
                    .font(.largeTitle.bold())
                    .multilineTextAlignment(.center)

                Text("Your AI-powered assistant for navigating the world with confidence.")
                    .font(.body)
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)

                Button(action: {
                    showControlsGuide = true
                }) {
                    Text("How Controls Work")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity, minHeight: 44)
                        .foregroundStyle(.white)
                        .background(actionButtonColor)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .buttonStyle(.plain)
                .accessibilityHint("Opens a quick guide for gestures, camera controls, and settings")

                Button(action: {
                    showProfileSetup = true
                }) {
                    Text("Get Started")
                        .font(.title3.bold())
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(actionButtonColor)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }
                .accessibilityLabel("Get started with SightLine")
                .accessibilityIdentifier("onboarding-get-started")

                Button(action: {
                    hasCompletedOnboarding = true
                }) {
                    Text("Skip for Now")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity, minHeight: 44)
                        .foregroundStyle(.primary)
                        .background(Color(uiColor: .secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.primary.opacity(0.14), lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Skip onboarding and go to main screen")
                .accessibilityIdentifier("onboarding-skip")
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 24)
            .frame(maxWidth: .infinity)
        }
        .background(Color(.systemBackground))
        .sheet(isPresented: $showProfileSetup) {
            UserProfileOnboardingView {
                hasCompletedOnboarding = true
            }
        }
        .sheet(isPresented: $showControlsGuide) {
            InteractionGuideView()
        }
    }
}

struct InteractionGuideView: View {
    @Environment(\.dismiss) private var dismiss

    private let tips = [
        ("Single tap", "Mute or unmute the microphone"),
        ("Double tap", "Interrupt speech immediately"),
        ("Triple tap", "Repeat the last response"),
        ("Long press", "Pause or resume the session"),
        ("Swipe up or down", "Change detail level"),
        ("Swipe left or right", "Turn the camera on or off"),
        ("Settings button", "Open profile, familiar faces, and user switching")
    ]

    var body: some View {
        NavigationStack {
            List {
                Section("Before You Start") {
                    Text("SightLine is gesture-driven. Review the main controls now or reopen this guide from the home screen.")
                        .foregroundStyle(.secondary)
                }

                Section("Core Controls") {
                    ForEach(tips, id: \.0) { title, description in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(title)
                                .font(.headline)
                            Text(description)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
            .navigationTitle("Controls Guide")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
