//
//  WeatherManager.swift
//  SightLine
//
//  Fetches current weather using WeatherKit and publishes a snapshot
//  for telemetry. Refreshes every 10 minutes to conserve API quota.
//

import WeatherKit
import CoreLocation
import Foundation
import Combine
import os

class WeatherManager: ObservableObject {
    @Published var currentWeather: WeatherSnapshot?
    @Published var isAvailable: Bool = true

    private static let logger = Logger(subsystem: "com.sightline.app", category: "Weather")
    private let weatherService = WeatherService.shared
    private var refreshTimer: Timer?
    private var cancellables = Set<AnyCancellable>()
    private static let refreshInterval: TimeInterval = 600  // 10 minutes

    struct WeatherSnapshot: Codable {
        var temperature: Double         // Celsius
        var condition: String           // "Rain", "Clear", "Snow" etc.
        var precipitation: String       // "none", "rain", "snow", "sleet", "hail", "mixed"
        var precipitationChance: Double // 0.0-1.0
        var windSpeed: Double           // m/s
        var visibility: Double          // meters
        var uvIndex: Int
        var humidity: Double            // 0.0-1.0
    }

    /// Start periodic weather monitoring using location updates.
    func startMonitoring(locationManager: LocationManager) {
        // Fetch immediately when we have a valid location
        locationManager.$latitude
            .combineLatest(locationManager.$longitude)
            .filter { lat, lng in lat != 0.0 || lng != 0.0 }
            .first()
            .sink { [weak self] lat, lng in
                let location = CLLocation(latitude: lat, longitude: lng)
                Task { await self?.fetchWeather(for: location) }
                self?.startRefreshTimer(locationManager: locationManager)
            }
            .store(in: &cancellables)
    }

    /// Fetch current weather for a location.
    func fetchWeather(for location: CLLocation) async {
        do {
            let weather = try await weatherService.weather(for: location)
            let current = weather.currentWeather

            let snapshot = WeatherSnapshot(
                temperature: current.temperature.converted(to: .celsius).value,
                condition: current.condition.description,
                precipitation: Self.mapPrecipitation(current.condition),
                precipitationChance: weather.hourlyForecast.first?.precipitationChance ?? 0.0,
                windSpeed: current.wind.speed.converted(to: .metersPerSecond).value,
                visibility: current.visibility.converted(to: .meters).value,
                uvIndex: current.uvIndex.value,
                humidity: current.humidity
            )

            await MainActor.run {
                self.currentWeather = snapshot
            }

            Self.logger.info("Weather updated: \(current.condition.description), \(current.temperature.converted(to: .celsius).value, format: .fixed(precision: 1))°C")
        } catch {
            Self.logger.error("Weather fetch failed: \(error.localizedDescription)")
            await MainActor.run {
                self.currentWeather = nil
                self.isAvailable = false
            }
        }
    }

    func stopMonitoring() {
        refreshTimer?.invalidate()
        refreshTimer = nil
        cancellables.removeAll()
    }

    // MARK: - Private

    private func startRefreshTimer(locationManager: LocationManager) {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(
            withTimeInterval: Self.refreshInterval,
            repeats: true
        ) { [weak self, weak locationManager] _ in
            guard let self, let lm = locationManager else { return }
            let lat = lm.latitude
            let lng = lm.longitude
            guard lat != 0.0 || lng != 0.0 else { return }
            let location = CLLocation(latitude: lat, longitude: lng)
            Task { await self.fetchWeather(for: location) }
        }
    }

    private static func mapPrecipitation(_ condition: WeatherCondition) -> String {
        switch condition {
        case .rain, .heavyRain, .drizzle:
            return "rain"
        case .snow, .heavySnow, .flurries, .blizzard:
            return "snow"
        case .sleet, .freezingRain, .freezingDrizzle:
            return "sleet"
        case .hail:
            return "hail"
        case .wintryMix:
            return "mixed"
        default:
            return "none"
        }
    }
}
