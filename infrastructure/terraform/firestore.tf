# =============================================================================
# SightLine — Firestore Configuration
# Native mode database for user profiles, face library, and memories
# =============================================================================

resource "google_firestore_database" "main" {
  project     = var.project_id
  name        = var.firestore_database_id
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  # Prevent accidental deletion
  deletion_policy = "DELETE"

  depends_on = [google_project_service.apis["firestore.googleapis.com"]]
}

# --- Firestore Vector Indexes -----------------------------------------------
# Required for face recognition (512-D) and memory search (2048-D)
# Per Infrastructure Report §1.4

# Face Library Vector Index (512-D, COSINE)
# Collection: users/{user_id}/face_library/{face_id}
resource "google_firestore_index" "face_library_vector" {
  project    = var.project_id
  database   = google_firestore_database.main.name
  collection = "face_library"

  fields {
    field_path = "embedding"
    vector_config {
      dimension = 512
      flat {}
    }
  }

  query_scope = "COLLECTION"

  depends_on = [google_firestore_database.main]
}

# Memories Vector Index (2048-D, COSINE)
# Collection: memories/{memory_id}
resource "google_firestore_index" "memories_vector" {
  project    = var.project_id
  database   = google_firestore_database.main.name
  collection = "memories"

  fields {
    field_path = "embedding"
    vector_config {
      dimension = 2048
      flat {}
    }
  }

  query_scope = "COLLECTION"

  depends_on = [google_firestore_database.main]
}
