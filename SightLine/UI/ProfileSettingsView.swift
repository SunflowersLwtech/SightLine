//
//  ProfileSettingsView.swift
//  SightLine
//
//  Unified settings page combining user profile editing, familiar faces
//  management, and user switching. Replaces the scattered quick-action
//  buttons that were unfriendly to visually impaired users.
//
//  Reuses: UserProfileModel, FaceRegistrationModel, UserSwitcherSheet.
//

import SwiftUI
import os

private let logger = Logger(subsystem: "com.sightline.app", category: "ProfileSettings")

struct ProfileSettingsView: View {
    @StateObject private var profileModel = UserProfileModel()
    @StateObject private var faceModel = FaceRegistrationModel()
    @ObservedObject var webSocketManager: WebSocketManager

    let onSwitchUser: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showUserSwitcher = false
    @State private var availableUsers: [String] = []
    @State private var showSaveBanner = false

    var body: some View {
        NavigationStack {
            Form {
                identitySection
                mobilitySection
                preferencesSection
                familiarFacesSection
                accountSection
                saveSection
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close") { dismiss() }
                        .accessibilityLabel("Close settings")
                }
            }
            .sheet(isPresented: $showUserSwitcher) {
                UserSwitcherSheet(
                    availableUsers: availableUsers,
                    currentUserId: SightLineConfig.defaultUserId,
                    onSelect: { userId in
                        showUserSwitcher = false
                        dismiss()
                        onSwitchUser(userId)
                    }
                )
            }
            .task {
                await profileModel.loadProfile()
                await faceModel.loadFaces()
            }
        }
    }

    // MARK: - Identity

    private var identitySection: some View {
        Section("Identity") {
            TextField("Preferred Name", text: $profileModel.preferredName)
                .textContentType(.name)
                .accessibilityLabel("Preferred name")

            Picker("Vision Status", selection: $profileModel.visionStatus) {
                Text("Totally Blind").tag("totally_blind")
                Text("Low Vision").tag("low_vision")
            }
            .accessibilityLabel("Vision status")

            Picker("Blindness Onset", selection: $profileModel.blindnessOnset) {
                Text("Congenital").tag("congenital")
                Text("Acquired").tag("acquired")
            }
            .accessibilityLabel("Blindness onset type")

            if profileModel.blindnessOnset == "acquired" {
                TextField("Age at Vision Loss", text: $profileModel.onsetAge)
                    .keyboardType(.numberPad)
                    .accessibilityLabel("Age when vision was lost")
            }
        }
    }

    // MARK: - Mobility

    private var mobilitySection: some View {
        Section("Mobility") {
            Toggle("Guide Dog", isOn: $profileModel.hasGuideDog)
                .accessibilityLabel("I use a guide dog")

            Toggle("White Cane", isOn: $profileModel.hasWhiteCane)
                .accessibilityLabel("I use a white cane")

            Picker("O&M Level", selection: $profileModel.omLevel) {
                Text("Beginner").tag("beginner")
                Text("Intermediate").tag("intermediate")
                Text("Advanced").tag("advanced")
            }
            .accessibilityLabel("Orientation and mobility experience level")

            Picker("Travel Frequency", selection: $profileModel.travelFrequency) {
                Text("Daily").tag("daily")
                Text("Weekly").tag("weekly")
                Text("Rarely").tag("rarely")
            }
            .accessibilityLabel("Independent travel frequency")
        }
    }

    // MARK: - Preferences

    private var preferencesSection: some View {
        Section("Preferences") {
            Picker("Verbosity", selection: $profileModel.verbosityPreference) {
                Text("Concise").tag("concise")
                Text("Detailed").tag("detailed")
            }
            .accessibilityLabel("Description verbosity preference")

            Picker("Description Priority", selection: $profileModel.descriptionPriority) {
                Text("Spatial Layout").tag("spatial")
                Text("Object Focus").tag("object")
            }
            .accessibilityLabel("Description priority")

            Toggle("Color Descriptions", isOn: $profileModel.colorDescription)
                .accessibilityLabel("Include color descriptions")

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Speech Speed")
                    Spacer()
                    Text(String(format: "%.1fx", profileModel.ttsSpeed))
                        .foregroundStyle(.secondary)
                }
                Slider(value: $profileModel.ttsSpeed, in: 0.8...2.5, step: 0.1)
                    .accessibilityLabel("Text-to-speech speed")
            }

            Picker("Language", selection: $profileModel.language) {
                Text("English (US)").tag("en-US")
                Text("English (UK)").tag("en-GB")
                Text("中文 (Chinese)").tag("zh-CN")
                Text("Español").tag("es-ES")
                Text("日本語").tag("ja-JP")
            }
            .accessibilityLabel("Preferred language")
        }
    }

    // MARK: - Familiar Faces

    private var familiarFacesSection: some View {
        Section("Familiar Faces") {
            if faceModel.isLoadingFaces {
                HStack {
                    ProgressView()
                    Text("Loading faces...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            } else if faceModel.registeredPersons.isEmpty {
                Text("No faces registered yet")
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("No familiar faces registered")
            } else {
                ForEach(faceModel.registeredPersons) { person in
                    HStack(spacing: 12) {
                        Image(systemName: "person.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.blue)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(person.personName)
                                .font(.body.bold())
                            Text("\(person.relationship.capitalized) · \(person.photoCount) photo\(person.photoCount == 1 ? "" : "s")")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("\(person.personName), \(person.relationship), \(person.photoCount) photos")
                }
                .onDelete { indexSet in
                    let persons = faceModel.registeredPersons
                    for index in indexSet {
                        let person = persons[index]
                        Task { await faceModel.deletePerson(person) }
                    }
                }
            }

            NavigationLink {
                FaceRegistrationView(isStandalone: false)
            } label: {
                Label("Add Familiar Face", systemImage: "person.badge.plus")
            }
            .accessibilityLabel("Add a new familiar face")
        }
    }

    // MARK: - Account

    private var accountSection: some View {
        Section("Account") {
            HStack {
                Text("User ID")
                Spacer()
                Text(SightLineConfig.defaultUserId)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Current user ID: \(SightLineConfig.defaultUserId)")

            Button {
                Task { await fetchUsers() }
                showUserSwitcher = true
            } label: {
                Label("Switch User", systemImage: "person.2.fill")
            }
            .accessibilityLabel("Switch to a different user profile")
        }
    }

    // MARK: - Save

    private var saveSection: some View {
        Section {
            Button {
                Task {
                    await profileModel.saveProfile()
                    if profileModel.saveSuccess {
                        webSocketManager.sendText("{\"type\":\"profile_updated\"}")
                        logger.info("Sent profile_updated notification via WebSocket")
                        showSaveBanner = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                            showSaveBanner = false
                        }
                    }
                }
            } label: {
                HStack {
                    Spacer()
                    if profileModel.isSaving {
                        ProgressView()
                            .padding(.trailing, 8)
                    }
                    Text(profileModel.isSaving ? "Saving..." : "Save Profile")
                        .fontWeight(.semibold)
                    Spacer()
                }
            }
            .disabled(profileModel.isSaving)
            .accessibilityLabel("Save profile to server")

            if showSaveBanner && profileModel.saveSuccess {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text(profileModel.saveResult)
                        .font(.subheadline)
                        .foregroundStyle(.green)
                }
                .accessibilityLabel("Success: \(profileModel.saveResult)")
            }

            if !profileModel.errorMessage.isEmpty {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text(profileModel.errorMessage)
                        .font(.subheadline)
                        .foregroundStyle(.red)
                }
                .accessibilityLabel("Error: \(profileModel.errorMessage)")
            }
        }
    }

    // MARK: - Helpers

    private func fetchUsers() async {
        let baseURL = SightLineConfig.serverBaseURL
            .replacingOccurrences(of: "wss://", with: "https://")
            .replacingOccurrences(of: "ws://", with: "http://")
        guard let url = URL(string: "\(baseURL)/api/users") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let users = json["users"] as? [String] {
                await MainActor.run { availableUsers = users }
            }
        } catch {
            logger.error("Failed to fetch users: \(error.localizedDescription)")
        }
    }
}
