# =============================================================================
# SightLine — Secret Manager Configuration
# Secure storage for API keys (Gemini + Google Maps)
# =============================================================================

# --- Gemini API Key ---------------------------------------------------------

resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

resource "google_secret_manager_secret_version" "gemini_api_key_version" {
  count = var.gemini_api_key != "" ? 1 : 0

  secret      = google_secret_manager_secret.gemini_api_key.id
  secret_data = var.gemini_api_key
}

# --- Google Maps API Key ----------------------------------------------------

resource "google_secret_manager_secret" "google_maps_api_key" {
  secret_id = "google-maps-api-key"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

resource "google_secret_manager_secret_version" "google_maps_api_key_version" {
  count = var.google_maps_api_key != "" ? 1 : 0

  secret      = google_secret_manager_secret.google_maps_api_key.id
  secret_data = var.google_maps_api_key
}
