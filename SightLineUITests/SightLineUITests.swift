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
    }

    private func launchApp(arguments: [String] = []) {
        app = XCUIApplication()
        app.launchArguments = arguments
        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    /// Verify the app launches without crashing.
    func testAppLaunches() throws {
        launchApp()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 10))
    }

    func testClosingProfileSetupKeepsOnboardingVisible() throws {
        launchApp(arguments: ["-uitest-reset-onboarding"])

        let getStarted = app.buttons["onboarding-get-started"]
        XCTAssertTrue(getStarted.waitForExistence(timeout: 5))
        getStarted.tap()

        let close = app.buttons["onboarding-profile-close"]
        XCTAssertTrue(close.waitForExistence(timeout: 5))
        close.tap()

        XCTAssertTrue(getStarted.waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["main-open-settings"].exists)
    }

    func testMainScreenExposesHelpAndSettingsButtons() throws {
        launchApp(arguments: ["-uitest-complete-onboarding"])

        XCTAssertTrue(app.buttons["main-open-guide"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["main-open-settings"].exists)
    }

    /// Verify the app performs a basic accessibility audit.
    @available(iOS 17.0, *)
    func testAccessibilityAudit() throws {
        launchApp(arguments: ["-uitest-reset-onboarding"])
        XCTAssertTrue(app.buttons["onboarding-get-started"].waitForExistence(timeout: 5))
        try app.performAccessibilityAudit()
    }

    func testDeniedMicrophoneShowsRecoveryButton() throws {
        launchApp(arguments: ["-uitest-complete-onboarding", "-uitest-mic-denied"])

        let settingsButton = app.buttons["main-open-app-settings"]
        XCTAssertTrue(settingsButton.waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["main-open-guide"].exists)
    }

    func testDeniedCameraDoesNotBlockMainScreenAtLaunch() throws {
        launchApp(arguments: [
            "-uitest-complete-onboarding",
            "-uitest-mic-granted",
            "-uitest-camera-denied"
        ])

        XCTAssertTrue(app.buttons["main-open-guide"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["main-open-app-settings"].exists)
    }
}
