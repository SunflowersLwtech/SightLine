# =============================================================================
# SightLine — Terraform Outputs
# Key deployment URLs and resource identifiers
# =============================================================================

output "cloud_run_url" {
  description = "Cloud Run service URL (use for iOS app WebSocket connection)"
  value       = google_cloud_run_v2_service.backend.uri
}

output "cloud_run_websocket_url" {
  description = "WebSocket endpoint for iOS app connection"
  value       = "wss://${replace(google_cloud_run_v2_service.backend.uri, "https://", "")}/ws/{user_id}/{session_id}"
}

output "service_account_email" {
  description = "Backend service account email"
  value       = google_service_account.backend.email
}

output "artifact_registry_url" {
  description = "Artifact Registry Docker image URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo}"
}

output "firestore_database" {
  description = "Firestore database name"
  value       = google_firestore_database.main.name
}

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "deployment_summary" {
  description = "Deployment summary for quick reference"
  value = <<-EOT
    ╔══════════════════════════════════════════════════════════════╗
    ║              SightLine Deployment Summary                    ║
    ║  AI-Powered Assistant for Visually Impaired Users            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Cloud Run URL:  ${google_cloud_run_v2_service.backend.uri}
    ║  Project:        ${var.project_id}
    ║  Region:         ${var.region}
    ║  Service Account: ${google_service_account.backend.email}
    ║  Firestore:      ${google_firestore_database.main.name}
    ╚══════════════════════════════════════════════════════════════╝
  EOT
}
