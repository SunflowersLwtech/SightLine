# =============================================================================
# SightLine — Cloud Run v2 Service Configuration
# Backend deployment with WebSocket support, Secret Manager integration,
# and optimized settings for real-time AI assistant
# =============================================================================

locals {
  # Use provided container image or build from Artifact Registry
  image_url = var.container_image != "" ? var.container_image : "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo}/${var.cloud_run_service_name}:latest"
}

resource "google_cloud_run_v2_service" "backend" {
  name     = var.cloud_run_service_name
  location = var.region
  project  = var.project_id

  # Allow external traffic (iOS app connects via WebSocket)
  ingress = "INGRESS_TRAFFIC_ALL"

  # Prevent accidental deletion
  deletion_protection = false

  template {
    # --- Service Account ---
    service_account = google_service_account.backend.email

    # --- Scaling ---
    scaling {
      min_instance_count = var.cloud_run_min_instances  # 1 = eliminate cold start
      max_instance_count = var.cloud_run_max_instances  # 10 = handle burst
    }

    # --- Timeout for WebSocket long connections ---
    timeout = var.cloud_run_timeout

    # --- CPU always allocated (no throttling) for WebSocket ---
    containers {
      image = local.image_url

      # --- Resource Limits ---
      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
        cpu_idle          = false  # no-cpu-throttling: keep CPU allocated
        startup_cpu_boost = true   # cpu-boost: faster cold starts
      }

      # --- Port ---
      ports {
        container_port = 8080
      }

      # --- Environment Variables ---
      dynamic "env" {
        for_each = var.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }

      # --- Secrets as Environment Variables ---
      # Gemini API Key from Secret Manager
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      # Google Maps API Key from Secret Manager
      env {
        name = "GOOGLE_MAPS_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.google_maps_api_key.secret_id
            version = "latest"
          }
        }
      }

      # --- Startup Probe ---
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 5
      }
    }
  }

  # --- Traffic: 100% to latest revision ---
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.apis["run.googleapis.com"],
    google_secret_manager_secret.gemini_api_key,
    google_secret_manager_secret.google_maps_api_key,
    google_service_account.backend,
    google_project_iam_member.backend_roles,
    google_secret_manager_secret_iam_member.gemini_key_access,
    google_secret_manager_secret_iam_member.maps_key_access,
  ]
}
