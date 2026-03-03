# =============================================================================
# SightLine — Terraform Main Configuration
# AI-powered real-time assistant for visually impaired users
#
# This IaC configuration provisions all GCP resources required by SightLine:
#   - 11 GCP APIs
#   - Firestore (Native mode)
#   - Secret Manager (Gemini + Maps API keys)
#   - Artifact Registry (Docker images)
#   - Cloud Run v2 (backend service)
#   - Service Account + IAM bindings
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Optional: remote state in GCS bucket
  # backend "gcs" {
  #   bucket = "sightline-terraform-state"
  #   prefix = "terraform/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Data Sources -----------------------------------------------------------

data "google_project" "current" {}

# --- GCP API Enablement -----------------------------------------------------
# All 11 APIs required by SightLine (per CLAUDE.md §GCP 已就绪的服务)

locals {
  required_apis = [
    "aiplatform.googleapis.com",         # Vertex AI
    "firestore.googleapis.com",          # Firestore
    "run.googleapis.com",                # Cloud Run
    "secretmanager.googleapis.com",      # Secret Manager
    "maps-backend.googleapis.com",       # Google Maps
    "places-backend.googleapis.com",     # Places API
    "geocoding-backend.googleapis.com",  # Geocoding
    "routes.googleapis.com",             # Routes API
    "cloudbuild.googleapis.com",         # Cloud Build
    "artifactregistry.googleapis.com",   # Artifact Registry
    "generativelanguage.googleapis.com", # Gemini Developer API
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.required_apis)

  project                    = var.project_id
  service                    = each.value
  disable_dependent_services = false
  disable_on_destroy         = false
}
