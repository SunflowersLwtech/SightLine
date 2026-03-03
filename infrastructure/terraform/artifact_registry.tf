# =============================================================================
# SightLine — Artifact Registry Configuration
# Docker image repository for the backend container
# =============================================================================

resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = var.artifact_registry_repo
  description   = "Docker images for SightLine backend - AI assistant for visually impaired users"
  format        = "DOCKER"
  project       = var.project_id

  # Cleanup policy: keep latest 5 images, delete untagged after 7 days
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"

    condition {
      tag_state = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  cleanup_policies {
    id     = "keep-latest"
    action = "KEEP"

    most_recent_versions {
      keep_count = 5
    }
  }

  depends_on = [google_project_service.apis["artifactregistry.googleapis.com"]]
}
