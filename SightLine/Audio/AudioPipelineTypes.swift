//
//  AudioPipelineTypes.swift
//  SightLine
//
//  Unified error vocabulary for the audio capture/playback pipeline.
//  Follows the existing CameraManager.onCameraFailure callback pattern.
//

import Foundation

enum AudioPipelineError: String {
    case engineNotRunning
    case formatCreationFailed
    case converterCreationFailed
    case tapInstallationFailed
    case bufferOverflow
    case playbackFormatFailed
    case vadModelNotFound
    case vadModelLoadFailed
    case vadUnavailable
}
