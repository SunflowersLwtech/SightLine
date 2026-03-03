//
//  LocationManager.swift
//  SightLine
//
//  Provides GPS location, heading, and space transition detection
//  using CLLocationManager. Feeds TelemetryAggregator.
//

import CoreLocation
import Foundation
import Combine
import os

class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published var latitude: Double = 0.0
    @Published var longitude: Double = 0.0
    @Published var accuracy: Double = 0.0
    @Published var speed: Double = 0.0
    @Published var altitude: Double = 0.0
    @Published var heading: Double = 0.0
    @Published var spaceTransitionDetected: Bool = false

    private static let logger = Logger(subsystem: "com.sightline.app", category: "Location")

    private let locationManager = CLLocationManager()
    private var lastAccuracy: Double = 0.0
    private var isMonitoring = false

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = 3.0
    }

    /// Request permission and start location + heading updates.
    func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true

        let status = locationManager.authorizationStatus
        if status == .notDetermined {
            locationManager.requestWhenInUseAuthorization()
        } else if status == .authorizedWhenInUse || status == .authorizedAlways {
            beginUpdates()
        }
    }

    /// Stop all location monitoring.
    func stopMonitoring() {
        guard isMonitoring else { return }
        isMonitoring = false
        locationManager.stopUpdatingLocation()
        locationManager.stopUpdatingHeading()
        Self.logger.info("Location monitoring stopped")
    }

    private func beginUpdates() {
        locationManager.startUpdatingLocation()
        if CLLocationManager.headingAvailable() {
            locationManager.startUpdatingHeading()
        }
        Self.logger.info("Location updates started")
    }

    // MARK: - CLLocationManagerDelegate

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        if (status == .authorizedWhenInUse || status == .authorizedAlways) && isMonitoring {
            beginUpdates()
        } else if status == .denied || status == .restricted {
            Self.logger.warning("Location permission denied")
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }

        let newAccuracy = location.horizontalAccuracy

        // Space transition detection:
        // Outdoor (<10m accuracy) -> Indoor (>30m accuracy) or vice versa
        if lastAccuracy > 0 {
            let wasOutdoor = lastAccuracy < 10
            let nowIndoor = newAccuracy > 30
            let wasIndoor = lastAccuracy > 30
            let nowOutdoor = newAccuracy < 10

            if (wasOutdoor && nowIndoor) || (wasIndoor && nowOutdoor) {
                DispatchQueue.main.async {
                    self.spaceTransitionDetected = true
                }
                Self.logger.info("Space transition detected: accuracy \(self.lastAccuracy)m → \(newAccuracy)m")
            }
        }
        lastAccuracy = newAccuracy

        DispatchQueue.main.async {
            self.latitude = location.coordinate.latitude
            self.longitude = location.coordinate.longitude
            self.accuracy = location.horizontalAccuracy
            self.speed = max(0, location.speed)
            self.altitude = location.altitude
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        guard newHeading.headingAccuracy >= 0 else { return }
        DispatchQueue.main.async {
            self.heading = newHeading.trueHeading
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        Self.logger.error("Location error: \(error.localizedDescription)")
    }

    /// Clear space transition flag after it has been sent in telemetry.
    func clearSpaceTransition() {
        spaceTransitionDetected = false
    }
}
