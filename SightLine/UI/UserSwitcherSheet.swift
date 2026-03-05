//
//  UserSwitcherSheet.swift
//  SightLine
//
//  Half-sheet for switching between demo user profiles.
//

import SwiftUI

struct UserSwitcherSheet: View {
    let availableUsers: [String]
    let isLoading: Bool
    let errorMessage: String
    let currentUserId: String
    let onRetry: () -> Void
    let onCancel: () -> Void
    let onSelect: (String) -> Void

    var body: some View {
        NavigationStack {
            List {
                if isLoading {
                    HStack(spacing: 12) {
                        ProgressView()
                        Text("Loading users...")
                            .foregroundStyle(.secondary)
                    }
                } else if !errorMessage.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text(errorMessage)
                            .foregroundStyle(.secondary)
                        Button("Try Again", action: onRetry)
                    }
                    .padding(.vertical, 8)
                } else if availableUsers.isEmpty {
                    Text("No users available")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(availableUsers, id: \.self) { userId in
                        Button {
                            onSelect(userId)
                        } label: {
                            HStack {
                                Text(userId)
                                    .foregroundStyle(.primary)
                                Spacer()
                                if userId == currentUserId {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(.blue)
                                }
                            }
                        }
                        .accessibilityLabel("\(userId)\(userId == currentUserId ? ", current user" : "")")
                    }
                }
            }
            .navigationTitle("Switch User")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Close", action: onCancel)
                }
            }
        }
        .presentationDetents([.medium])
    }
}
