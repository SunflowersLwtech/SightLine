#!/usr/bin/env bash
# =============================================================================
# SightLine — One-Click Deployment Script
# Deploys the complete infrastructure and backend service to GCP
#
# Usage:
#   ./deploy.sh                    # Full deployment (infra + build + deploy)
#   ./deploy.sh --infra-only       # Terraform infrastructure only
#   ./deploy.sh --build-only       # Docker build + push only
#   ./deploy.sh --plan             # Terraform plan (dry run)
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${SCRIPT_DIR}/terraform"

# GCP defaults (override via env vars or terraform.tfvars)
PROJECT_ID="${PROJECT_ID:-sightline-hackathon}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-sightline-backend}"
AR_REPO="${AR_REPO:-sightline}"

IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}"

# --- Colors ------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Helper Functions --------------------------------------------------------

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[✓]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
log_error()   { echo -e "${RED}[✗]${NC} $*"; }

banner() {
  echo -e "${CYAN}"
  echo "╔═══════════════════════════════════════════════════════════════╗"
  echo "║          SightLine — Automated GCP Deployment                 ║"
  echo "║  AI-Powered Real-Time Assistant for Visually Impaired Users   ║"
  echo "╚═══════════════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

# --- Prerequisite Check ------------------------------------------------------

check_prerequisites() {
  log_info "Checking prerequisites..."
  local missing=0

  # Terraform
  if ! command -v terraform &>/dev/null; then
    log_error "terraform not found. Install: brew install terraform"
    missing=1
  else
    log_success "terraform $(terraform version -json 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["terraform_version"])' 2>/dev/null || terraform version | head -1)"
  fi

  # gcloud
  if ! command -v gcloud &>/dev/null; then
    log_error "gcloud not found. Install: https://cloud.google.com/sdk/install"
    missing=1
  else
    log_success "gcloud $(gcloud version 2>/dev/null | head -1)"
  fi

  # docker
  if ! command -v docker &>/dev/null; then
    log_error "docker not found. Install: https://docker.com/get-started"
    missing=1
  else
    log_success "docker $(docker --version 2>/dev/null)"
  fi

  if [ $missing -ne 0 ]; then
    log_error "Missing prerequisites. Please install the above tools and retry."
    exit 1
  fi

  # Check gcloud auth
  if ! gcloud auth print-access-token &>/dev/null; then
    log_warn "Not authenticated with gcloud. Running: gcloud auth login"
    gcloud auth login
  fi

  # Set project
  gcloud config set project "${PROJECT_ID}" 2>/dev/null
  log_success "GCP project set to: ${PROJECT_ID}"
}

# --- Docker Build & Push -----------------------------------------------------

build_and_push() {
  log_info "Building and pushing Docker image..."

  # Configure Docker for Artifact Registry
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

  # Build
  log_info "Building image: ${IMAGE_URL}:latest"
  docker build -t "${IMAGE_URL}:latest" "${PROJECT_ROOT}"

  # Tag with git short hash for traceability
  GIT_SHA=$(cd "${PROJECT_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  docker tag "${IMAGE_URL}:latest" "${IMAGE_URL}:${GIT_SHA}"

  # Push
  log_info "Pushing image..."
  docker push "${IMAGE_URL}:latest"
  docker push "${IMAGE_URL}:${GIT_SHA}"

  log_success "Image pushed: ${IMAGE_URL}:latest (${GIT_SHA})"
}

# --- Terraform Infrastructure ------------------------------------------------

terraform_init() {
  log_info "Initializing Terraform..."
  cd "${TF_DIR}"

  terraform init -upgrade
  log_success "Terraform initialized"
}

terraform_plan() {
  log_info "Running Terraform plan..."
  cd "${TF_DIR}"

  terraform plan \
    -var="container_image=${IMAGE_URL}:latest" \
    -out=tfplan

  log_success "Terraform plan completed. Review the plan above."
}

terraform_apply() {
  log_info "Applying Terraform configuration..."
  cd "${TF_DIR}"

  if [ -f tfplan ]; then
    terraform apply tfplan
  else
    terraform apply \
      -var="container_image=${IMAGE_URL}:latest" \
      -auto-approve
  fi

  log_success "Terraform apply completed!"

  # Show outputs
  echo ""
  terraform output deployment_summary 2>/dev/null || true
}

# --- Post-Deployment Verification --------------------------------------------

verify_deployment() {
  log_info "Verifying deployment..."

  # Get Cloud Run URL
  CLOUD_RUN_URL=$(cd "${TF_DIR}" && terraform output -raw cloud_run_url 2>/dev/null)

  if [ -n "${CLOUD_RUN_URL}" ]; then
    log_info "Testing health endpoint: ${CLOUD_RUN_URL}/health"
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${CLOUD_RUN_URL}/health" --max-time 30 || echo "000")

    if [ "${HTTP_STATUS}" = "200" ]; then
      log_success "Health check passed! (HTTP ${HTTP_STATUS})"
    else
      log_warn "Health check returned HTTP ${HTTP_STATUS} (service may still be starting)"
    fi

    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Deployment Complete!                                      ║${NC}"
    echo -e "${GREEN}║                                                            ║${NC}"
    echo -e "${GREEN}║  Service URL: ${CLOUD_RUN_URL}${NC}"
    echo -e "${GREEN}║  WebSocket:   wss://${CLOUD_RUN_URL#https://}/ws/{user_id}/{session_id}${NC}"
    echo -e "${GREEN}║                                                            ║${NC}"
    echo -e "${GREEN}║  Update iOS Config.swift with the URL above.               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
  else
    log_warn "Could not retrieve Cloud Run URL. Check Terraform outputs."
  fi
}

# --- Main Execution ----------------------------------------------------------

main() {
  banner

  case "${1:-full}" in
    --plan)
      check_prerequisites
      terraform_init
      terraform_plan
      ;;
    --infra-only)
      check_prerequisites
      terraform_init
      terraform_plan
      terraform_apply
      ;;
    --build-only)
      check_prerequisites
      build_and_push
      ;;
    full|--full|"")
      check_prerequisites
      build_and_push
      terraform_init
      terraform_plan
      terraform_apply
      verify_deployment
      ;;
    --help|-h)
      echo "Usage: $0 [OPTION]"
      echo ""
      echo "Options:"
      echo "  (none)         Full deployment (build + infra + verify)"
      echo "  --plan         Terraform plan only (dry run)"
      echo "  --infra-only   Terraform apply only (no Docker build)"
      echo "  --build-only   Docker build + push only (no Terraform)"
      echo "  --help         Show this help message"
      echo ""
      echo "Environment variables:"
      echo "  PROJECT_ID     GCP project ID (default: sightline-hackathon)"
      echo "  REGION         GCP region (default: us-central1)"
      echo "  SERVICE_NAME   Cloud Run service name (default: sightline-backend)"
      echo "  AR_REPO        Artifact Registry repo (default: sightline)"
      ;;
    *)
      log_error "Unknown option: $1"
      echo "Run '$0 --help' for usage information."
      exit 1
      ;;
  esac
}

main "$@"
