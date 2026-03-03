//
//  WatchHeadingManager.swift
//  SightLineWatch
//
//  Reads compass heading on watchOS via CLLocationManager.
//  Requires location permission (NSLocationWhenInUseUsageDescription).
//

import CoreLocation
import Foundation
import Combine
import os

/// Reads compass heading on watchOS. Requires location permission.
class WatchHeadingManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published var heading: Double?
    @Published var headingAccuracy: Double?

    private static let logger = Logger(subsystem: "com.sightline.watch", category: "Heading")
    private let locationManager = CLLocationManager()

    override init() {
        super.init()
        locationManager.delegate = self
    }

    func startUpdates() {
        guard CLLocationManager.headingAvailable() else {
            Self.logger.warning("Heading not available on this device")
            return
        }
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingHeading()
        Self.logger.info("Heading updates started")
    }

    func stopUpdates() {
        locationManager.stopUpdatingHeading()
        Self.logger.info("Heading updates stopped")
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let h = newHeading.magneticHeading
        guard h >= 0 else { return }  // -1 means invalid
        heading = h
        headingAccuracy = newHeading.headingAccuracy
    }
}
