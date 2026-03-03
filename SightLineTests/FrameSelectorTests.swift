//
//  FrameSelectorTests.swift
//  SightLineTests
//
//  Tests for LOD-based frame throttling logic.
//

import Testing
import Foundation
@testable import SightLine

@Suite("FrameSelector LOD Throttling")
struct FrameSelectorTests {

    @Test("LOD 1 interval is 1.0s (safety mode)")
    func lod1Interval() {
        let selector = FrameSelector()
        selector.currentLOD = 1
        #expect(selector.minInterval == 1.0)
    }

    @Test("LOD 2 interval is 1.0s (balanced mode)")
    func lod2Interval() {
        let selector = FrameSelector()
        selector.currentLOD = 2
        #expect(selector.minInterval == 1.0)
    }

    @Test("LOD 3 interval is 2.0s (exploration mode)")
    func lod3Interval() {
        let selector = FrameSelector()
        selector.currentLOD = 3
        #expect(selector.minInterval == 2.0)
    }

    @Test("unknown LOD defaults to 1.0s")
    func unknownLodDefault() {
        let selector = FrameSelector()
        selector.currentLOD = 99
        #expect(selector.minInterval == 1.0)
    }

    @Test("initial state allows first frame immediately")
    func initialStateAllowsFrame() {
        let selector = FrameSelector()
        #expect(selector.shouldSendFrame())
    }

    @Test("markFrameSent blocks immediate next frame")
    func markFrameSentBlocksNext() {
        let selector = FrameSelector()
        selector.currentLOD = 2
        selector.markFrameSent()
        #expect(!selector.shouldSendFrame())
    }

    @Test("default LOD is 2")
    func defaultLOD() {
        let selector = FrameSelector()
        #expect(selector.currentLOD == 2)
    }
}
