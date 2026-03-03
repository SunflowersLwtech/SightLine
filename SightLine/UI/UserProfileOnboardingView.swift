//
//  UserProfileOnboardingView.swift
//  SightLine
//
//  Onboarding flow for new users to set up their vision profile.
//  Collects vision_status, blindness_onset, mobility aids, verbosity
//  preferences, and personalization options to drive adaptive descriptions.
//
//  Spec: Final_Specification.md §2.4, Consolidated_Reference.md §2.2
//  Backend: POST /api/profile/{user_id}, GET /api/profile/{user_id}
//
//  Designed for accessibility: all controls have VoiceOver labels,
//  large touch targets, and clear semantic grouping.
//

import SwiftUI
import Combine
import os

private let logger = Logger(subsystem: "com.sightline.app", category: "UserProfile")

// MARK: - Profile Model

@MainActor
final class UserProfileModel: ObservableObject {
    // Core vision info
    @Published var visionStatus: String = "totally_blind"
    @Published var blindnessOnset: String = "congenital"
    @Published var onsetAge: String = ""  // String for TextField, convert to Int
    @Published var preferredName: String = ""

    // Mobility aids
    @Published var hasGuideDog: Bool = false
    @Published var hasWhiteCane: Bool = false

    // Preferences
    @Published var verbosityPreference: String = "concise"
    @Published var language: String = "en-US"
    @Published var descriptionPriority: String = "spatial"
    @Published var colorDescription: Bool = false
    @Published var ttsSpeed: Double = 1.5
    @Published var omLevel: String = "intermediate"
    @Published var travelFrequency: String = "daily"

    // UI state
    @Published var currentStep: Int = 0
    @Published var isSaving: Bool = false
    @Published var isLoading: Bool = false
    @Published var saveResult: String = ""
    @Published var saveSuccess: Bool = false
    @Published var errorMessage: String = ""
    @Published var hasExistingProfile: Bool = false

    let totalSteps = 4

    private var baseURL: String {
        let wsURL = SightLineConfig.serverBaseURL
        return wsURL
            .replacingOccurrences(of: "wss://", with: "https://")
            .replacingOccurrences(of: "ws://", with: "http://")
    }

    // MARK: - Load Existing Profile

    func loadProfile() async {
        isLoading = true
        let userId = SightLineConfig.defaultUserId

        guard let url = URL(string: "\(baseURL)/api/profile/\(userId)") else {
            isLoading = false
            return
        }

        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse else {
                isLoading = false
                return
            }

            if httpResponse.statusCode == 200 {
                let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
                populateFromJSON(json)
                hasExistingProfile = true
                logger.info("Loaded existing profile for user \(userId)")
            } else {
                hasExistingProfile = false
                logger.info("No existing profile for user \(userId)")
            }
        } catch {
            logger.error("Load profile failed: \(error)")
        }

        isLoading = false
    }

    private func populateFromJSON(_ json: [String: Any]) {
        visionStatus = json["vision_status"] as? String ?? "totally_blind"
        blindnessOnset = json["blindness_onset"] as? String ?? "congenital"
        if let age = json["onset_age"] as? Int {
            onsetAge = "\(age)"
        }
        preferredName = json["preferred_name"] as? String ?? ""
        hasGuideDog = json["has_guide_dog"] as? Bool ?? false
        hasWhiteCane = json["has_white_cane"] as? Bool ?? false
        verbosityPreference = json["verbosity_preference"] as? String ?? "concise"
        language = json["language"] as? String ?? "en-US"
        descriptionPriority = json["description_priority"] as? String ?? "spatial"
        colorDescription = json["color_description"] as? Bool ?? false
        ttsSpeed = json["tts_speed"] as? Double ?? 1.5
        omLevel = json["om_level"] as? String ?? "intermediate"
        travelFrequency = json["travel_frequency"] as? String ?? "daily"
    }

    // MARK: - Save Profile

    func saveProfile() async {
        isSaving = true
        errorMessage = ""

        let userId = SightLineConfig.defaultUserId
        guard let url = URL(string: "\(baseURL)/api/profile/\(userId)") else {
            errorMessage = "Invalid server URL"
            isSaving = false
            return
        }

        var body: [String: Any] = [
            "vision_status": visionStatus,
            "blindness_onset": blindnessOnset,
            "has_guide_dog": hasGuideDog,
            "has_white_cane": hasWhiteCane,
            "tts_speed": ttsSpeed,
            "verbosity_preference": verbosityPreference,
            "language": language,
            "description_priority": descriptionPriority,
            "color_description": colorDescription,
            "om_level": omLevel,
            "travel_frequency": travelFrequency,
        ]

        if !preferredName.trimmingCharacters(in: .whitespaces).isEmpty {
            body["preferred_name"] = preferredName.trimmingCharacters(in: .whitespaces)
        }
        if let age = Int(onsetAge), age > 0 {
            body["onset_age"] = age
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 15

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                errorMessage = "Invalid server response"
                isSaving = false
                return
            }

            if httpResponse.statusCode == 200 {
                saveResult = "Profile saved successfully!"
                saveSuccess = true
                hasExistingProfile = true
                logger.info("Profile saved for user \(userId)")
            } else {
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                let error = json?["error"] as? String ?? "Unknown error"
                errorMessage = "Save failed: \(error)"
                saveSuccess = false
            }
        } catch {
            errorMessage = "Network error: \(error.localizedDescription)"
            saveSuccess = false
        }

        isSaving = false
    }
}

// MARK: - Onboarding View

struct UserProfileOnboardingView: View {
    @StateObject private var model = UserProfileModel()
    @Environment(\.dismiss) private var dismiss
    @State private var showQuickPresets = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Step indicators
                stepIndicator

                // Step content
                TabView(selection: $model.currentStep) {
                    step1VisionStatus.tag(0)
                    step2MobilityAids.tag(1)
                    step3Preferences.tag(2)
                    step4Review.tag(3)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .animation(.easeInOut(duration: 0.3), value: model.currentStep)

                // Navigation buttons
                navigationButtons
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle(model.hasExistingProfile ? "Edit Profile" : "Welcome to SightLine")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Presets") { showQuickPresets = true }
                        .font(.subheadline)
                }
            }
            .sheet(isPresented: $showQuickPresets) {
                presetsSheet
            }
            .task {
                await model.loadProfile()
            }
        }
    }

    // MARK: - Step Indicator

    private var stepIndicator: some View {
        let stepNames = ["Vision", "Mobility", "Preferences", "Review"]
        return HStack(spacing: 8) {
            ForEach(0..<model.totalSteps, id: \.self) { step in
                Capsule()
                    .fill(step <= model.currentStep ? Color.blue : Color.gray.opacity(0.3))
                    .frame(height: 4)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Step \(model.currentStep + 1) of \(model.totalSteps): \(stepNames[model.currentStep])")
        .accessibilityValue("\(model.currentStep + 1) of \(model.totalSteps) completed")
    }

    // MARK: - Step 1: Vision Status

    private var step1VisionStatus: some View {
        ScrollView {
            VStack(spacing: 24) {
                stepHeader(
                    icon: "eye.slash",
                    title: "Vision Information",
                    subtitle: "Help SightLine understand your vision to personalize descriptions."
                )

                // Preferred name
                formField(label: "Your Name") {
                    TextField("How should SightLine call you?", text: $model.preferredName)
                        .textFieldStyle(.roundedBorder)
                        .textContentType(.name)
                        .accessibilityLabel("Preferred name")
                }

                // Vision status
                formField(label: "Vision Status") {
                    Picker("Vision Status", selection: $model.visionStatus) {
                        Text("Totally Blind").tag("totally_blind")
                        Text("Low Vision").tag("low_vision")
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Vision status")
                }

                // Blindness onset
                formField(label: "When did vision loss begin?") {
                    Picker("Onset", selection: $model.blindnessOnset) {
                        Text("Born Blind (Congenital)").tag("congenital")
                        Text("Acquired Later").tag("acquired")
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Blindness onset type")
                }

                // Onset age (only if acquired)
                if model.blindnessOnset == "acquired" {
                    formField(label: "Age at Vision Loss") {
                        TextField("e.g. 25", text: $model.onsetAge)
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                            .accessibilityLabel("Age when vision was lost")
                    }
                }

                explanationCard(
                    "Why this matters",
                    text: "People born blind may not have visual memory of colors or objects. SightLine uses this to choose whether to describe colors, use visual analogies, or focus on spatial/tactile descriptions."
                )
            }
            .padding()
        }
    }

    // MARK: - Step 2: Mobility Aids

    private var step2MobilityAids: some View {
        ScrollView {
            VStack(spacing: 24) {
                stepHeader(
                    icon: "figure.walk",
                    title: "Mobility & Navigation",
                    subtitle: "Tell us about your mobility tools and experience."
                )

                formField(label: "Mobility Aids") {
                    VStack(spacing: 12) {
                        Toggle(isOn: $model.hasGuideDog) {
                            HStack {
                                Image(systemName: "dog")
                                Text("Guide Dog")
                            }
                        }
                        .accessibilityLabel("I use a guide dog")

                        Toggle(isOn: $model.hasWhiteCane) {
                            HStack {
                                Image(systemName: "figure.walk")
                                Text("White Cane")
                            }
                        }
                        .accessibilityLabel("I use a white cane")
                    }
                }

                formField(label: "Orientation & Mobility Level") {
                    Picker("O&M Level", selection: $model.omLevel) {
                        Text("Beginner").tag("beginner")
                        Text("Intermediate").tag("intermediate")
                        Text("Advanced").tag("advanced")
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Orientation and mobility experience level")
                }

                formField(label: "How often do you travel independently?") {
                    Picker("Travel Frequency", selection: $model.travelFrequency) {
                        Text("Daily").tag("daily")
                        Text("Weekly").tag("weekly")
                        Text("Rarely").tag("rarely")
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Independent travel frequency")
                }

                explanationCard(
                    "How this helps",
                    text: "Experienced navigators need less verbose directions. SightLine adjusts its Level of Detail based on your mobility experience."
                )
            }
            .padding()
        }
    }

    // MARK: - Step 3: Preferences

    private var step3Preferences: some View {
        ScrollView {
            VStack(spacing: 24) {
                stepHeader(
                    icon: "slider.horizontal.3",
                    title: "Description Preferences",
                    subtitle: "Customize how SightLine describes the world to you."
                )

                formField(label: "Verbosity") {
                    Picker("Verbosity", selection: $model.verbosityPreference) {
                        Text("Concise").tag("concise")
                        Text("Detailed").tag("detailed")
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Description verbosity preference")
                }

                formField(label: "Description Priority") {
                    Picker("Priority", selection: $model.descriptionPriority) {
                        Text("Spatial Layout").tag("spatial")
                        Text("Object Focus").tag("object")
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Whether to prioritize spatial layout or object descriptions")
                }

                formField(label: "Include Color Descriptions") {
                    Toggle(isOn: $model.colorDescription) {
                        Text(model.colorDescription ? "Colors will be described" : "No color descriptions")
                            .font(.subheadline)
                    }
                    .accessibilityLabel("Include color descriptions in scene analysis")
                }

                formField(label: "Speech Speed") {
                    VStack(spacing: 4) {
                        Slider(value: $model.ttsSpeed, in: 0.8...2.5, step: 0.1)
                            .accessibilityLabel("Text-to-speech speed")
                        HStack {
                            Text("Slower")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(String(format: "%.1fx", model.ttsSpeed))
                                .font(.caption.bold())
                                .foregroundStyle(.blue)
                            Spacer()
                            Text("Faster")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                formField(label: "Language") {
                    Picker("Language", selection: $model.language) {
                        Text("English (US)").tag("en-US")
                        Text("English (UK)").tag("en-GB")
                        Text("中文 (Chinese)").tag("zh-CN")
                        Text("Español").tag("es-ES")
                        Text("日本語").tag("ja-JP")
                    }
                    .accessibilityLabel("Preferred language")
                }
            }
            .padding()
        }
    }

    // MARK: - Step 4: Review & Save

    private var step4Review: some View {
        ScrollView {
            VStack(spacing: 24) {
                stepHeader(
                    icon: "checkmark.circle",
                    title: "Review Your Profile",
                    subtitle: "Confirm your settings. You can change these anytime."
                )

                VStack(alignment: .leading, spacing: 12) {
                    reviewRow("Name", value: model.preferredName.isEmpty ? "(not set)" : model.preferredName)
                    reviewRow("Vision", value: model.visionStatus == "totally_blind" ? "Totally Blind" : "Low Vision")
                    reviewRow("Onset", value: model.blindnessOnset == "congenital" ? "Congenital" : "Acquired\(model.onsetAge.isEmpty ? "" : " (age \(model.onsetAge))")")
                    reviewRow("Guide Dog", value: model.hasGuideDog ? "Yes" : "No")
                    reviewRow("White Cane", value: model.hasWhiteCane ? "Yes" : "No")
                    reviewRow("O&M Level", value: model.omLevel.capitalized)
                    reviewRow("Verbosity", value: model.verbosityPreference.capitalized)
                    reviewRow("Priority", value: model.descriptionPriority == "spatial" ? "Spatial Layout" : "Object Focus")
                    reviewRow("Colors", value: model.colorDescription ? "Included" : "Excluded")
                    reviewRow("Speed", value: String(format: "%.1fx", model.ttsSpeed))
                    reviewRow("Language", value: model.language)
                }
                .padding()
                .background(Color(.systemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))

                if !model.errorMessage.isEmpty {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.yellow)
                        Text(model.errorMessage)
                            .font(.subheadline)
                            .foregroundStyle(.red)
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.red.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Error: \(model.errorMessage)")
                    .accessibilityAddTraits(.updatesFrequently)
                }

                if model.saveSuccess {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text(model.saveResult)
                            .font(.subheadline)
                            .foregroundStyle(.green)
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Success: \(model.saveResult)")
                    .accessibilityAddTraits(.updatesFrequently)
                }
            }
            .padding()
        }
    }

    // MARK: - Navigation Buttons

    private var navigationButtons: some View {
        HStack(spacing: 16) {
            if model.currentStep > 0 {
                Button(action: { model.currentStep -= 1 }) {
                    HStack {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color(.systemGray5))
                    .foregroundStyle(.primary)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .accessibilityLabel("Go to previous step")
            }

            if model.currentStep < model.totalSteps - 1 {
                Button(action: { model.currentStep += 1 }) {
                    HStack {
                        Text("Next")
                        Image(systemName: "chevron.right")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.blue)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .accessibilityLabel("Go to next step")
            } else {
                Button(action: {
                    Task {
                        await model.saveProfile()
                        if model.saveSuccess {
                            try? await Task.sleep(nanoseconds: 1_500_000_000)
                            dismiss()
                        }
                    }
                }) {
                    HStack(spacing: 8) {
                        if model.isSaving {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Image(systemName: "checkmark.circle.fill")
                        }
                        Text(model.isSaving ? "Saving..." : "Save Profile")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(model.isSaving ? Color.gray : Color.green)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .disabled(model.isSaving)
                .accessibilityLabel("Save profile to server")
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(Color(.systemGroupedBackground))
    }

    // MARK: - Quick Presets Sheet

    private var presetsSheet: some View {
        NavigationStack {
            List {
                Section("Quick Presets") {
                    presetButton(
                        name: "Congenital Blind",
                        description: "Born blind, concise descriptions, spatial focus, no colors",
                        icon: "eye.slash.fill"
                    ) {
                        model.visionStatus = "totally_blind"
                        model.blindnessOnset = "congenital"
                        model.hasGuideDog = true
                        model.hasWhiteCane = false
                        model.ttsSpeed = 2.0
                        model.verbosityPreference = "concise"
                        model.descriptionPriority = "spatial"
                        model.colorDescription = false
                        model.omLevel = "intermediate"
                        model.travelFrequency = "daily"
                    }

                    presetButton(
                        name: "Acquired Low Vision",
                        description: "Lost vision later, detailed descriptions, colors included",
                        icon: "eye.trianglebadge.exclamationmark"
                    ) {
                        model.visionStatus = "low_vision"
                        model.blindnessOnset = "acquired"
                        model.onsetAge = "30"
                        model.hasGuideDog = false
                        model.hasWhiteCane = true
                        model.ttsSpeed = 1.3
                        model.verbosityPreference = "detailed"
                        model.descriptionPriority = "object"
                        model.colorDescription = true
                        model.omLevel = "intermediate"
                        model.travelFrequency = "weekly"
                    }

                    presetButton(
                        name: "Beginner Low Vision",
                        description: "Recently impaired, very detailed, learning navigation",
                        icon: "figure.walk.motion"
                    ) {
                        model.visionStatus = "low_vision"
                        model.blindnessOnset = "acquired"
                        model.onsetAge = "45"
                        model.hasGuideDog = false
                        model.hasWhiteCane = false
                        model.ttsSpeed = 1.2
                        model.verbosityPreference = "detailed"
                        model.descriptionPriority = "object"
                        model.colorDescription = true
                        model.omLevel = "beginner"
                        model.travelFrequency = "rarely"
                    }
                }
            }
            .navigationTitle("Presets")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showQuickPresets = false }
                }
            }
        }
        .presentationDetents([.medium])
    }

    // MARK: - Reusable Components

    private func stepHeader(icon: String, title: String, subtitle: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 40))
                .foregroundStyle(.blue)

            Text(title)
                .font(.title3.bold())

            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.bottom, 8)
    }

    private func formField<Content: View>(label: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
            content()
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func reviewRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.subheadline.bold())
        }
    }

    private func explanationCard(_ title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.blue)
                Text(title)
                    .font(.subheadline.bold())
                    .foregroundStyle(.blue)
            }
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(Color.blue.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func presetButton(name: String, description: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: {
            action()
            showQuickPresets = false
        }) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundStyle(.blue)
                    .frame(width: 40)

                VStack(alignment: .leading, spacing: 2) {
                    Text(name)
                        .font(.body.bold())
                        .foregroundStyle(.primary)
                    Text(description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 4)
        }
    }
}
