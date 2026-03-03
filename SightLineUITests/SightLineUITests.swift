//
//  SightLineUITests.swift
//  SightLineUITests
//
//  UI Tests for SightLine iOS app.
//  Validates app launch, basic accessibility, and core UI elements.
//

import XCTest

final class SightLineUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    /// Verify the app launches without crashing.
    func testAppLaunches() throws {
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 10))
    }

    /// Verify the app performs a basic accessibility audit.
    @available(iOS 17.0, *)
    func testAccessibilityAudit() throws {
        try app.performAccessibilityAudit()
    }
}
