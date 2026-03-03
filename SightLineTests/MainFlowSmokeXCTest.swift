//
//  MainFlowSmokeXCTest.swift
//  SightLineTests
//
//  XCTest smoke checks for core app protocol flow.
//

import XCTest
@testable import SightLine

final class MainFlowSmokeXCTest: XCTestCase {

    func testGestureSerialization() {
        let json = UpstreamMessage.gesture(type: "sos").toJSON()
        XCTAssertEqual(json, "{\"type\":\"gesture\",\"gesture\":\"sos\"}")
    }

    func testSessionReadyAndLodParse() {
        let sessionReady = DownstreamMessage.parse(text: "{\"type\":\"session_ready\"}")
        if case .sessionReady? = sessionReady {
            // pass
        } else {
            XCTFail("Expected .sessionReady")
        }

        let lodUpdate = DownstreamMessage.parse(text: "{\"type\":\"lod_update\",\"lod\":1}")
        if case .lodUpdate(let lod)? = lodUpdate {
            XCTAssertEqual(lod, 1)
        } else {
            XCTFail("Expected .lodUpdate")
        }
    }

}
