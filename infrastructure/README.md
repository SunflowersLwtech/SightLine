# SightLine Infrastructure — Terraform

## Overview

This directory contains Infrastructure as Code (IaC) configuration for the SightLine project, an AI-powered real-time assistant for visually impaired users.

All GCP resources are managed via Terraform, enabling reproducible, one-command deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Google Cloud Platform                         │
│                        Project: sightline-hackathon                  │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────────────────────┐      │
│  │  Cloud Run v2     │     │  Firestore (Native)              │      │
│  │  sightline-backend│────▶│  • users/{uid}/face_library      │      │
│  │  • 2 vCPU, 2Gi   │     │  • memories (Vector 2048-D)      │      │
│  │  • min 1 instance │     │  • face_library (Vector 512-D)   │      │
│  │  • WebSocket 3600s│     └──────────────────────────────────┘      │
│  └──────┬───────────┘                                                │
│         │                  ┌──────────────────────────────────┐      │
│         │                  │  Secret Manager                   │      │
│         ├─────────────────▶│  • gemini-api-key                │      │
│         │                  │  • google-maps-api-key            │      │
│         │                  └──────────────────────────────────┘      │
│         │                                                            │
│         │                  ┌──────────────────────────────────┐      │
│         │                  │  Artifact Registry                │      │
│         └─────────────────▶│  • sightline (Docker)            │      │
│                            └──────────────────────────────────┘      │
│                                                                      │
│  Service Account: sightline-backend@sightline-hackathon.iam.gsa.com │
│  Roles: datastore.user, aiplatform.user, secretmanager.secretAccessor│
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Terraform** >= 1.5.0: `brew install terraform`
2. **Google Cloud SDK**: `brew install google-cloud-sdk`
3. **Docker**: `brew install --cask docker`
4. **GCP Project** with billing enabled
5. **API Keys** already generated (Gemini + Google Maps)

## Quick Start

```bash
# 1. Navigate to infrastructure directory
cd SightLine/infrastructure

# 2. Copy and edit variables
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
# Edit terraform.tfvars with your API keys

# 3. Deploy everything (Docker build + Terraform apply)
chmod +x deploy.sh
./deploy.sh

# 4. Or just preview changes
./deploy.sh --plan
```

## Files

```
infrastructure/
├── deploy.sh                       # One-click deployment script
├── destroy.sh                      # Infrastructure teardown script
├── README.md                       # This file
└── terraform/
    ├── main.tf                     # Provider + 11 GCP API enablement
    ├── variables.tf                # All configurable parameters
    ├── outputs.tf                  # Deployment URLs and summary
    ├── cloud_run.tf                # Cloud Run v2 service
    ├── firestore.tf                # Firestore + vector indexes
    ├── secrets.tf                  # Secret Manager
    ├── iam.tf                      # Service Account + IAM bindings
    ├── artifact_registry.tf        # Docker image repository
    └── terraform.tfvars.example    # Example variable values
```

## Deployment Modes

| Command | Description |
|---------|-------------|
| `./deploy.sh` | Full deployment (build + infra + verify) |
| `./deploy.sh --plan` | Terraform plan only (dry run) |
| `./deploy.sh --infra-only` | Infrastructure only (skip Docker build) |
| `./deploy.sh --build-only` | Docker build + push only |
| `./destroy.sh` | Tear down all infrastructure |

## Resources Created

| Resource | Terraform Resource | Description |
|----------|-------------------|-------------|
| 11 GCP APIs | `google_project_service` | All required APIs |
| Cloud Run v2 | `google_cloud_run_v2_service` | Backend service |
| Firestore DB | `google_firestore_database` | Native mode database |
| Face Vector Index | `google_firestore_index` | 512-D InsightFace |
| Memory Vector Index | `google_firestore_index` | 2048-D Gemini Embedding |
| Secret Manager (x2) | `google_secret_manager_secret` | API keys |
| Artifact Registry | `google_artifact_registry_repository` | Docker images |
| Service Account | `google_service_account` | Backend identity |
| IAM Bindings (x4) | `google_project_iam_member` | Role assignments |

## After Deployment

1. Update **iOS Config.swift** with the Cloud Run URL from Terraform output
2. Run **seed scripts** to populate demo data:
   ```bash
   cd SightLine
   python scripts/seed_firestore.py
   python scripts/seed_user_profile.py
   ```
3. Verify WebSocket connection from iOS simulator
