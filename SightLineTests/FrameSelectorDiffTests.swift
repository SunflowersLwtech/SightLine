//
//  FrameSelectorDiffTests.swift
//  SightLineTests
//
//  Tests pixel-diff frame deduplication with real JPEG inputs.
//

import Testing
import UIKit
@testable import SightLine

@Suite("Frame Selector Pixel Diff")
struct FrameSelectorDiffTests {

    @Test("first frame is always considered different")
    func firstFrameAlwaysDifferent() {
        let selector = FrameSelector()
        let red = makeSolidJPEG(color: .red)

        #expect(selector.isFrameDifferent(jpegData: red))
    }

    @Test("identical scene frame is skipped by pixel diff")
    func identicalFrameSkipped() {
        let selector = FrameSelector()
        let redA = makeSolidJPEG(color: .red)
        let redB = makeSolidJPEG(color: .red)

        #expect(selector.isFrameDifferent(jpegData: redA))
        #expect(!selector.isFrameDifferent(jpegData: redB))
    }

    @Test("changed scene frame is sent by pixel diff")
    func changedFrameSent() {
        let selector = FrameSelector()
        let dark = makeSolidJPEG(color: .black)
        let bright = makeSolidJPEG(color: .white)

        #expect(selector.isFrameDifferent(jpegData: dark))
        #expect(selector.isFrameDifferent(jpegData: bright))
    }

    @Test("invalid JPEG data defaults to send")
    func invalidJpegDefaultsToSend() {
        let selector = FrameSelector()
        let invalid = Data([0x00, 0x11, 0x22, 0x33])

        #expect(selector.isFrameDifferent(jpegData: invalid))
    }

    private func makeSolidJPEG(color: UIColor) -> Data {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 64, height: 64))
        let image = renderer.image { context in
            color.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        }
        return image.jpegData(compressionQuality: 0.95)!
    }
}
