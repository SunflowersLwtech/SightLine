# =============================================================================
# SightLine Terraform Variables
# AI-powered assistant for visually impaired users
# =============================================================================

# --- GCP Project -----------------------------------------------------------

variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "sightline-hackathon"
}

variable "region" {
  description = "GCP Region for all resources"
  type        = string
  default     = "us-central1"
}

# --- Service Account -------------------------------------------------------

variable "service_account_id" {
  description = "Service Account ID for the backend"
  type        = string
  default     = "sightline-backend"
}

# --- Cloud Run --------------------------------------------------------------

variable "cloud_run_service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "sightline-backend"
}

variable "cloud_run_min_instances" {
  description = "Minimum number of Cloud Run instances (1 = eliminate cold start)"
  type        = number
  default     = 1
}

variable "cloud_run_max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "cloud_run_memory" {
  description = "Memory allocation per Cloud Run instance"
  type        = string
  default     = "2Gi"
}

variable "cloud_run_cpu" {
  description = "CPU allocation per Cloud Run instance"
  type        = string
  default     = "2"
}

variable "cloud_run_timeout" {
  description = "Request timeout in seconds (3600 for WebSocket long connections)"
  type        = string
  default     = "3600s"
}

variable "container_image" {
  description = "Full container image URL. If empty, uses Artifact Registry default."
  type        = string
  default     = ""
}

# --- Secrets ----------------------------------------------------------------

variable "gemini_api_key" {
  description = "Gemini API Key (stored in Secret Manager)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "google_maps_api_key" {
  description = "Google Maps API Key (stored in Secret Manager)"
  type        = string
  sensitive   = true
  default     = ""
}

# --- Artifact Registry ------------------------------------------------------

variable "artifact_registry_repo" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "sightline"
}

# --- Firestore ---------------------------------------------------------------

variable "firestore_database_id" {
  description = "Firestore database ID. Use '(default)' for the default database."
  type        = string
  default     = "(default)"
}

# --- Environment Variables ---------------------------------------------------

variable "env_vars" {
  description = "Additional environment variables for Cloud Run"
  type        = map(string)
  default = {
    GOOGLE_GENAI_USE_VERTEXAI = "TRUE"
    GOOGLE_CLOUD_LOCATION     = "us-central1"
  }
}
