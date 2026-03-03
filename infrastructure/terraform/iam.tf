# =============================================================================
# SightLine — IAM Configuration
# Service Account + Role Bindings
# =============================================================================

# --- Service Account --------------------------------------------------------

resource "google_service_account" "backend" {
  account_id   = var.service_account_id
  display_name = "SightLine Backend"
  description  = "Service account for SightLine Cloud Run backend - AI assistant for visually impaired users"
  project      = var.project_id
}

# --- IAM Role Bindings ------------------------------------------------------
# Per Infrastructure Report §4.3

locals {
  sa_roles = [
    "roles/datastore.user",               # Firestore read/write
    "roles/aiplatform.user",              # Vertex AI (Live API, Embedding, Memory Bank)
    "roles/secretmanager.secretAccessor",  # Secret Manager read
    "roles/run.invoker",                  # Cloud Run invocation
  ]
}

resource "google_project_iam_member" "backend_roles" {
  for_each = toset(local.sa_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# --- Secret Manager IAM for Service Account ---------------------------------
# Grant the backend SA access to read secrets

resource "google_secret_manager_secret_iam_member" "gemini_key_access" {
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.backend.email}"

  depends_on = [google_secret_manager_secret.gemini_api_key]
}

resource "google_secret_manager_secret_iam_member" "maps_key_access" {
  secret_id = google_secret_manager_secret.google_maps_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.backend.email}"

  depends_on = [google_secret_manager_secret.google_maps_api_key]
}

# --- Cloud Run Public Access ------------------------------------------------
# Allow unauthenticated access (iOS app connects directly)

resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"

  depends_on = [google_cloud_run_v2_service.backend]
}
