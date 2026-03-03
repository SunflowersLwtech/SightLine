//
//  UserSwitcherSheet.swift
//  SightLine
//
//  Half-sheet for switching between demo user profiles.
//

import SwiftUI

struct UserSwitcherSheet: View {
    let availableUsers: [String]
    let currentUserId: String
    let onSelect: (String) -> Void

    var body: some View {
        NavigationStack {
            List {
                if availableUsers.isEmpty {
                    Text("Loading users...")
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
        }
        .presentationDetents([.medium])
    }
}
