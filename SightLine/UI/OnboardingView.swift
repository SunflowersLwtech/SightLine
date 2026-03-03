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
    @Binding var hasCompletedOnboarding: Bool
    @State private var showProfileSetup = false

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Image(systemName: "eye.slash.circle.fill")
                .font(.system(size: 80))
                .foregroundStyle(.blue)
                .accessibilityHidden(true)

            Text("SightLine")
                .font(.largeTitle.bold())

            Text("Your AI-powered assistant for navigating the world with confidence.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer()

            Button(action: {
                showProfileSetup = true
            }) {
                Text("Get Started")
                    .font(.title3.bold())
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Color.blue)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .padding(.horizontal, 32)
            .accessibilityLabel("Get started with SightLine")

            Button("Skip for Now") {
                hasCompletedOnboarding = true
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .padding(.bottom, 24)
            .accessibilityLabel("Skip onboarding and go to main screen")
        }
        .background(Color(.systemBackground))
        .sheet(isPresented: $showProfileSetup, onDismiss: {
            hasCompletedOnboarding = true
        }) {
            UserProfileOnboardingView()
        }
    }
}
