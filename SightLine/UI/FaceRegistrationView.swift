//
//  FaceRegistrationView.swift
//  SightLine
//
//  Family/caregiver face registration UI.
//  Supports direct camera capture + photo library upload (3-5 samples),
//  explicit consent, and batched registration via REST.
//
//  Backend: POST /api/face/register, GET /api/face/list/{user_id}
//

import SwiftUI
import AVFoundation
import PhotosUI
import Combine
import os
import UIKit
@preconcurrency import Vision

private let logger = Logger(subsystem: "com.sightline.app", category: "FaceRegistration")

extension Notification.Name {
    static let faceLibraryChanged = Notification.Name("com.sightline.faceLibraryChanged")
}

// MARK: - Face Registration Model

@MainActor
final class FaceRegistrationModel: ObservableObject {
    struct RegisteredFace: Identifiable {
        let id: String     // face_id
        let personName: String
        let relationship: String
        let photoIndex: Int
        let createdAt: String
        let hasReferencePhoto: Bool
        let consentConfirmed: Bool
    }

    /// A person with one or more registered face embeddings, grouped by name.
    struct RegisteredPerson: Identifiable {
        let id: String          // person_name (unique key)
        let personName: String
        let relationship: String
        let photoCount: Int
        let faces: [RegisteredFace]
    }

    let minFaceSamples = 1
    let maxFaceSamples = 5

    @Published var personName: String = ""
    @Published var relationship: String = "friend"
    @Published var consentConfirmed: Bool = false

    @Published var isRegistering: Bool = false
    @Published var registrationResult: String = ""
    @Published var registrationSuccess: Bool = false
    @Published var registrationProgress: String = ""
    @Published var errorMessage: String = ""

    @Published var selectedImages: [UIImage] = []
    @Published var showCamera: Bool = false
    @Published var detectionStatus: String = ""

    @Published var registeredFaces: [RegisteredFace] = []
    @Published var isLoadingFaces: Bool = false

    /// Faces grouped by person name for display.
    var registeredPersons: [RegisteredPerson] {
        let grouped = Dictionary(grouping: registeredFaces) { $0.personName }
        return grouped.map { name, faces in
            RegisteredPerson(
                id: name,
                personName: name,
                relationship: faces.first?.relationship ?? "",
                photoCount: faces.count,
                faces: faces
            )
        }
        .sorted { $0.personName.localizedCaseInsensitiveCompare($1.personName) == .orderedAscending }
    }

    let relationships = ["friend", "family", "spouse", "colleague", "caregiver", "other"]

    var canRegister: Bool {
        !personName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        && selectedImages.count >= minFaceSamples
        && consentConfirmed
        && !isRegistering
    }

    private var baseURL: String {
        // Use the same server URL as the WebSocket, but with HTTPS.
        let wsURL = SightLineConfig.serverBaseURL
        return wsURL
            .replacingOccurrences(of: "wss://", with: "https://")
            .replacingOccurrences(of: "ws://", with: "http://")
    }

    // MARK: - Photo Draft Management

    func addSelectedImage(_ image: UIImage) {
        guard selectedImages.count < maxFaceSamples else {
            errorMessage = "You can upload up to \(maxFaceSamples) photos per person"
            return
        }
        selectedImages.append(image)
        registrationSuccess = false
        registrationResult = ""
        errorMessage = ""
        detectFaces(in: image, photoIndex: selectedImages.count)
    }

    /// Client-side face detection using Vision framework for immediate feedback.
    private func detectFaces(in image: UIImage, photoIndex: Int) {
        guard let cgImage = image.cgImage else {
            detectionStatus = "Photo \(photoIndex): Unable to process image"
            return
        }
        let request = VNDetectFaceRectanglesRequest { [weak self] request, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if let error = error {
                    self.detectionStatus = "Photo \(photoIndex): Detection error — \(error.localizedDescription)"
                    return
                }
                let faceCount = request.results?.count ?? 0
                switch faceCount {
                case 0:
                    self.detectionStatus = "Photo \(photoIndex): No face detected — try a clearer photo"
                case 1:
                    self.detectionStatus = "Photo \(photoIndex): 1 face detected"
                default:
                    self.detectionStatus = "Photo \(photoIndex): \(faceCount) faces detected — use a photo with one face"
                }
            }
        }
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }
    }

    func removeSelectedImage(at index: Int) {
        guard selectedImages.indices.contains(index) else { return }
        selectedImages.remove(at: index)
    }

    private func clearDraftAfterSuccess() {
        selectedImages.removeAll()
        personName = ""
        consentConfirmed = false
    }

    // MARK: - Register Face Samples

    func registerFaceSamples() async {
        let trimmedName = personName.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedName.isEmpty else {
            errorMessage = "Please enter the person's name"
            return
        }

        guard selectedImages.count >= minFaceSamples else {
            errorMessage = "Please upload at least \(minFaceSamples) photos"
            return
        }

        guard consentConfirmed else {
            errorMessage = "Please confirm consent before uploading photos"
            return
        }

        guard let url = URL(string: "\(baseURL)/api/face/register") else {
            errorMessage = "Invalid server URL"
            return
        }

        errorMessage = ""
        registrationResult = ""
        registrationSuccess = false
        registrationProgress = "Preparing upload..."
        isRegistering = true

        var successCount = 0
        var failures: [String] = []

        for (index, image) in selectedImages.enumerated() {
            registrationProgress = "Uploading photo \(index + 1) of \(selectedImages.count)..."

            guard let base64Image = encodeImageForUpload(image) else {
                failures.append("Photo \(index + 1): Failed to encode image")
                continue
            }

            let body: [String: Any] = [
                "user_id": SightLineConfig.defaultUserId,
                "person_name": trimmedName,
                "relationship": relationship,
                "image_base64": base64Image,
                "photo_index": index,
                "consent_confirmed": consentConfirmed,
                "store_reference_photo": true,
            ]

            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 45

            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
                let (data, response) = try await URLSession.shared.data(for: request)

                guard let httpResponse = response as? HTTPURLResponse else {
                    failures.append("Photo \(index + 1): Invalid server response")
                    continue
                }

                let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
                if httpResponse.statusCode == 201 {
                    successCount += 1
                } else {
                    let serverError = json["error"] as? String ?? "Unknown error"
                    failures.append("Photo \(index + 1): \(serverError)")
                }
            } catch {
                failures.append("Photo \(index + 1): \(error.localizedDescription)")
            }
        }

        isRegistering = false
        registrationProgress = ""

        if successCount >= 1 {
            registrationSuccess = true
            registrationResult = "Registered \(trimmedName) with \(successCount) photo(s)."
            logger.info("Face registration completed: \(trimmedName), success=\(successCount)")

            if !failures.isEmpty {
                errorMessage = "Some photos failed. \(failures.prefix(2).joined(separator: " | "))"
            }

            clearDraftAfterSuccess()
            NotificationCenter.default.post(name: .faceLibraryChanged, object: nil)
            await loadFaces()
            return
        }

        registrationSuccess = false
        errorMessage = "Registration failed. \(failures.prefix(2).joined(separator: " | "))"
        logger.error("Face registration failed: success=\(successCount), failures=\(failures.count)")
    }

    private func encodeImageForUpload(_ image: UIImage) -> String? {
        let prepared = downsample(image: image, maxEdge: 1280)
        guard let jpegData = prepared.jpegData(compressionQuality: 0.82) else {
            return nil
        }
        return jpegData.base64EncodedString()
    }

    private func downsample(image: UIImage, maxEdge: CGFloat) -> UIImage {
        let size = image.size
        let longestEdge = max(size.width, size.height)
        guard longestEdge > maxEdge, longestEdge > 0 else { return image }

        let scale = maxEdge / longestEdge
        let targetSize = CGSize(width: floor(size.width * scale), height: floor(size.height * scale))

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    // MARK: - Load Faces

    func loadFaces() async {
        isLoadingFaces = true
        let userId = SightLineConfig.defaultUserId

        guard let url = URL(string: "\(baseURL)/api/face/list/\(userId)") else {
            isLoadingFaces = false
            return
        }

        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                isLoadingFaces = false
                return
            }

            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            let faces = json["faces"] as? [[String: Any]] ?? []

            registeredFaces = faces.map { face in
                RegisteredFace(
                    id: face["face_id"] as? String ?? UUID().uuidString,
                    personName: face["person_name"] as? String ?? "Unknown",
                    relationship: face["relationship"] as? String ?? "",
                    photoIndex: face["photo_index"] as? Int ?? 0,
                    createdAt: face["created_at"] as? String ?? "",
                    hasReferencePhoto: face["has_reference_photo"] as? Bool ?? false,
                    consentConfirmed: face["consent_confirmed"] as? Bool ?? false
                )
            }
            logger.info("Loaded \(self.registeredFaces.count) registered face(s)")
        } catch {
            logger.error("Load faces failed: \(error.localizedDescription)")
        }

        isLoadingFaces = false
    }

    // MARK: - Delete Face

    func deleteFace(_ face: RegisteredFace) async {
        let userId = SightLineConfig.defaultUserId
        guard let url = URL(string: "\(baseURL)/api/face/\(userId)/\(face.id)") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                logger.info("Deleted face: \(face.personName) (\(face.id))")
                await loadFaces()
            }
        } catch {
            logger.error("Delete face failed: \(error.localizedDescription)")
        }
    }

    /// Delete all face entries for a person (all their embeddings).
    func deletePerson(_ person: RegisteredPerson) async {
        let userId = SightLineConfig.defaultUserId
        guard let encodedName = person.personName
            .addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "\(baseURL)/api/face/\(userId)?person_name=\(encodedName)") else {
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                logger.info("Deleted person: \(person.personName) (\(person.photoCount) photos)")
                NotificationCenter.default.post(name: .faceLibraryChanged, object: nil)
                await loadFaces()
            }
        } catch {
            logger.error("Delete person failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - Camera Capture View (UIImagePickerController wrapper)

struct CameraCaptureView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraDevice = .front
        picker.allowsEditing = false
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraCaptureView

        init(_ parent: CameraCaptureView) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage {
                parent.image = image
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Face Registration View

struct FaceRegistrationView: View {
    /// When `true` (default), wraps content in its own NavigationStack with a Close button.
    /// When `false`, omits NavigationStack — suitable for NavigationLink push from a parent.
    var isStandalone: Bool = true

    @StateObject private var model = FaceRegistrationModel()
    @Environment(\.dismiss) private var dismiss

    @State private var selectedPhotoItems: [PhotosPickerItem] = []
    @State private var latestCameraImage: UIImage?
    @State private var isImportingFromLibrary = false

    private var remainingSlots: Int {
        max(0, model.maxFaceSamples - model.selectedImages.count)
    }

    var body: some View {
        if isStandalone {
            NavigationStack {
                content
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Close") { dismiss() }
                        }
                    }
            }
        } else {
            content
        }
    }

    private var content: some View {
        ScrollView {
            VStack(spacing: 20) {
                instructionHeader

                photoSection

                formSection

                consentSection

                registerButton

                if !model.registrationProgress.isEmpty {
                    progressBanner
                }

                if !model.errorMessage.isEmpty {
                    errorBanner
                }

                if model.registrationSuccess {
                    successBanner
                }

                registeredFacesSection
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Register Familiar Faces")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $model.showCamera) {
            CameraCaptureView(image: $latestCameraImage)
        }
        .onChange(of: latestCameraImage) { _, newImage in
            guard let newImage else { return }
            model.addSelectedImage(newImage)
            latestCameraImage = nil
        }
        .onChange(of: selectedPhotoItems) { _, newItems in
            guard !newItems.isEmpty else { return }
            Task { await importPhotoPickerItems(newItems) }
        }
        .task {
            await model.loadFaces()
        }
    }

    // MARK: - Subviews

    private var instructionHeader: some View {
        VStack(spacing: 8) {
            Image(systemName: "person.crop.circle.badge.plus")
                .font(.system(size: 48))
                .foregroundStyle(.blue)

            Text("Upload Familiar Faces")
                .font(.title2.bold())

            Text("Add 1-5 clear photos for each person. More photos with different angles improve recognition accuracy.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding(.top, 8)
    }

    private var photoSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Photos")
                    .font(.headline)

                Spacer()

                Text("\(model.selectedImages.count)/\(model.maxFaceSamples)")
                    .font(.subheadline.bold())
                    .foregroundStyle(model.selectedImages.count >= model.minFaceSamples ? .green : .orange)
            }

            Text("At least 1 photo is required (up to 5). Use different angles and lighting for better recognition.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Button {
                    model.showCamera = true
                } label: {
                    Label("Take Photo", systemImage: "camera.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isRegistering || model.selectedImages.count >= model.maxFaceSamples)
                .accessibilityLabel("Take a new face photo")

                PhotosPicker(
                    selection: $selectedPhotoItems,
                    maxSelectionCount: max(1, remainingSlots),
                    matching: .images
                ) {
                    Label("Choose Photos", systemImage: "photo.on.rectangle.angled")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(model.isRegistering || model.selectedImages.count >= model.maxFaceSamples)
                .accessibilityLabel("Choose photos from library")
            }

            if isImportingFromLibrary {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Importing selected photos...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            if model.selectedImages.isEmpty {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.systemGray6))
                    .frame(height: 120)
                    .overlay(
                        Text("No photos selected yet")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    )
                    .accessibilityLabel("No selected photos")
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(Array(model.selectedImages.enumerated()), id: \.offset) { index, image in
                            ZStack(alignment: .topTrailing) {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 110, height: 110)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                                    )

                                Button(role: .destructive) {
                                    model.removeSelectedImage(at: index)
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.title3)
                                        .foregroundStyle(.white, .red)
                                }
                                .offset(x: 6, y: -6)
                                .accessibilityLabel("Remove photo \(index + 1)")
                            }
                            .accessibilityElement(children: .combine)
                            .accessibilityLabel("Selected photo \(index + 1)")
                        }
                    }
                    .padding(.vertical, 4)
                }
            }

            if !model.detectionStatus.isEmpty {
                Text(model.detectionStatus)
                    .font(.caption)
                    .foregroundStyle(model.detectionStatus.contains("No face") || model.detectionStatus.contains("faces detected")
                        ? .orange : .green)
                    .accessibilityLabel("Face detection: \(model.detectionStatus)")
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var formSection: some View {
        VStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Person's Name")
                    .font(.subheadline.bold())
                    .foregroundStyle(.secondary)

                TextField("e.g. Mom, David Chen", text: $model.personName)
                    .textFieldStyle(.roundedBorder)
                    .textContentType(.name)
                    .autocorrectionDisabled()
                    .accessibilityLabel("Person's name")
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Relationship")
                    .font(.subheadline.bold())
                    .foregroundStyle(.secondary)

                Picker("Relationship", selection: $model.relationship) {
                    ForEach(model.relationships, id: \.self) { relationship in
                        Text(relationship.capitalized).tag(relationship)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityLabel("Relationship to user")
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var consentSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Consent")
                .font(.headline)

            Toggle(isOn: $model.consentConfirmed) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("I confirm I have permission to upload and store these photos.")
                        .font(.subheadline)
                    Text("Stored photos are used for familiar-face recognition and can be deleted at any time from this screen.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .accessibilityLabel("Consent to upload and store familiar face photos")
            .accessibilityHint("Required before registration")
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var registerButton: some View {
        Button(action: {
            Task { await model.registerFaceSamples() }
        }) {
            HStack(spacing: 8) {
                if model.isRegistering {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Image(systemName: "icloud.and.arrow.up")
                }

                Text(model.isRegistering ? "Registering..." : "Upload and Register Faces")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(model.canRegister ? Color.blue : Color.gray)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .disabled(!model.canRegister)
        .accessibilityLabel("Upload selected photos and register faces")
    }

    private var progressBanner: some View {
        HStack(spacing: 8) {
            ProgressView()
            Text(model.registrationProgress)
                .font(.subheadline)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.blue.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var errorBanner: some View {
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

    private var successBanner: some View {
        HStack {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Text(model.registrationResult)
                .font(.subheadline)
                .foregroundStyle(.green)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.green.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Success: \(model.registrationResult)")
        .accessibilityAddTraits(.updatesFrequently)
    }

    private var registeredFacesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Registered Faces")
                    .font(.headline)

                Spacer()

                if model.isLoadingFaces {
                    ProgressView()
                }

                Button(action: {
                    Task { await model.loadFaces() }
                }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.subheadline)
                }
                .accessibilityLabel("Refresh face list")
            }

            if model.registeredPersons.isEmpty && !model.isLoadingFaces {
                Text("No faces registered yet")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 20)
            } else {
                ForEach(model.registeredPersons) { person in
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

                        Spacer()

                        Button(role: .destructive) {
                            Task { await model.deletePerson(person) }
                        } label: {
                            Image(systemName: "trash")
                                .font(.subheadline)
                        }
                        .accessibilityLabel("Delete \(person.personName)")
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .background(Color(.systemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Helpers

    @MainActor
    private func importPhotoPickerItems(_ items: [PhotosPickerItem]) async {
        isImportingFromLibrary = true
        defer {
            isImportingFromLibrary = false
            selectedPhotoItems = []
        }

        for item in items {
            if model.selectedImages.count >= model.maxFaceSamples {
                break
            }
            do {
                if let data = try await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    model.addSelectedImage(image)
                }
            } catch {
                model.errorMessage = "Failed to import one of the selected photos"
            }
        }
    }
}
