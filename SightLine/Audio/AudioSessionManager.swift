//
//  AudioSessionManager.swift
//  SightLine
//
//  Configures AVAudioSession for simultaneous recording and playback
//  with low-latency settings suitable for real-time voice conversation.
//

import AVFoundation

class AudioSessionManager {
    static let shared = AudioSessionManager()

    private init() {}

    func configure() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: .voiceChat,
            options: [
                .defaultToSpeaker,
                .allowBluetoothHFP
            ]
        )
        try session.setPreferredSampleRate(SightLineConfig.audioInputSampleRate)
        try session.setPreferredIOBufferDuration(0.01) // 10ms for low latency
        try session.setActive(true)
    }
}
