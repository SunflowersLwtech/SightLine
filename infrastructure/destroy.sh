#!/usr/bin/env bash
# =============================================================================
# SightLine — Infrastructure Teardown Script
# Destroys all Terraform-managed GCP resources
#
# ⚠️  WARNING: This will permanently delete all infrastructure resources!
#     Firestore data will NOT be deleted (data retention safety).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="${SCRIPT_DIR}/terraform"

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${RED}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ⚠️  WARNING: INFRASTRUCTURE TEARDOWN                        ║"
echo "║  This will destroy ALL Terraform-managed GCP resources!       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

read -p "Are you absolutely sure? Type 'destroy' to confirm: " confirmation

if [ "${confirmation}" != "destroy" ]; then
  echo "Aborted."
  exit 0
fi

cd "${TF_DIR}"

echo -e "${YELLOW}[!] Running Terraform destroy...${NC}"
terraform destroy -auto-approve

echo -e "${GREEN}[✓] Infrastructure destroyed.${NC}"
echo ""
echo "Note: Firestore data may still exist. Clean up manually if needed:"
echo "  gcloud firestore databases delete --database='(default)'"
