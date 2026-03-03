#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_DIR="${PROJECT_ROOT}/build"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SCHEME="${SIGHTLINE_WATCH_SCHEME:-SightLineWatch}"
WATCH_ARCH="${SIGHTLINE_WATCH_ARCH:-arm64}"
WATCH_DESTINATION_ID="${SIGHTLINE_WATCH_DESTINATION_ID:-}"
WATCH_BUNDLE_ID="${SIGHTLINE_WATCH_BUNDLE_ID:-com.sunflowers.SightLine.watchkitapp}"
ALLOW_PROVISIONING_UPDATES="${SIGHTLINE_ALLOW_PROVISIONING_UPDATES:-1}"
RETRY_ON_CLOCK_BLOCK="${SIGHTLINE_RETRY_ON_CLOCK_BLOCK:-1}"

mkdir -p "${RESULT_DIR}"

require_tool() {
  local tool="$1"
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Missing required tool: ${tool}"
    exit 2
  fi
}

resolve_watch_destination() {
  local destinations line
  destinations="$(cd "${PROJECT_ROOT}" && xcodebuild -scheme "${SCHEME}" -showdestinations 2>/dev/null || true)"
  line="$(printf '%s\n' "${destinations}" | rg -m1 "platform:watchOS, arch:${WATCH_ARCH}, id:[^,]+, name:" || true)"

  if [[ -z "${line}" ]]; then
    line="$(printf '%s\n' "${destinations}" | rg -m1 "platform:watchOS, arch:[^,]+, id:[^,]+, name:" || true)"
  fi

  if [[ -z "${line}" ]]; then
    echo "Unable to resolve a watchOS destination for scheme '${SCHEME}'."
    printf '%s\n' "${destinations}"
    exit 3
  fi

  WATCH_DESTINATION_ID="$(printf '%s\n' "${line}" | sed -E 's/.*id:([^,}]+).*/\1/')"
  if [[ -z "${WATCH_DESTINATION_ID}" ]]; then
    echo "Failed to parse watch destination id from: ${line}"
    exit 3
  fi
}

check_lock_state() {
  local lock_output
  if ! lock_output="$(xcrun devicectl device info lockState --device "${WATCH_DESTINATION_ID}" 2>&1)"; then
    echo "Warning: unable to query watch lock state; continuing."
    printf '%s\n' "${lock_output}"
    return 0
  fi

  printf '%s\n' "${lock_output}"
  if printf '%s\n' "${lock_output}" | rg -q "passcodeRequired: true|unlockedSinceBoot: false"; then
    echo "Watch is locked or has not been unlocked since boot."
    echo "Please unlock the watch once, keep it awake, then re-run this script."
    return 10
  fi
  return 0
}

launch_watch_foreground() {
  # Keep the watch app in a runnable foreground state before XCTest bootstraps.
  xcrun devicectl device process launch \
    --device "${WATCH_DESTINATION_ID}" \
    --activate \
    --terminate-existing \
    "${WATCH_BUNDLE_ID}" >/dev/null 2>&1 || true
}

run_test_attempt() {
  local attempt="$1"
  local result_bundle="${RESULT_DIR}/watch-device-test-${TIMESTAMP}-attempt${attempt}.xcresult"
  local log_file="${RESULT_DIR}/watch-device-test-${TIMESTAMP}-attempt${attempt}.log"

  local -a cmd=(
    xcodebuild test
    -scheme "${SCHEME}"
    -destination "platform=watchOS,id=${WATCH_DESTINATION_ID},arch=${WATCH_ARCH}"
    -resultBundlePath "${result_bundle}"
  )

  if [[ "${ALLOW_PROVISIONING_UPDATES}" == "1" ]]; then
    cmd+=(-allowProvisioningUpdates)
  fi

  echo "[Attempt ${attempt}] Running: ${cmd[*]}"
  set +e
  (cd "${PROJECT_ROOT}" && "${cmd[@]}") 2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  LAST_LOG_FILE="${log_file}"
  LAST_RESULT_BUNDLE="${result_bundle}"
  return "${status}"
}

print_clock_state_guidance() {
  echo
  echo "watchOS blocked app activation from clock:"
  echo "  Navigation away from clock is not allowed due to one or more active system states"
  echo
  echo "Do these on the watch, then retry:"
  echo "  1) Wake and unlock the watch (screen on, not asleep)."
  echo "  2) Exit Now Playing / Siri / phone-call UI / Wallet / passcode prompt."
  echo "  3) Ensure no system modal is covering the watch face."
  echo "  4) Re-run: scripts/run_watch_device_tests.sh"
}

require_tool xcodebuild
require_tool xcrun
require_tool rg

if [[ -z "${WATCH_DESTINATION_ID}" ]]; then
  resolve_watch_destination
fi

echo "Using watch destination id: ${WATCH_DESTINATION_ID}"
if ! check_lock_state; then
  exit 10
fi

launch_watch_foreground
sleep 2

if run_test_attempt 1; then
  echo "Watch device tests passed."
  exit 0
fi

if [[ "${RETRY_ON_CLOCK_BLOCK}" == "1" ]] && rg -q "Navigation away from clock is not allowed due to one or more active system states" "${LAST_LOG_FILE}"; then
  echo "Detected clock-navigation block. Retrying once after re-activating app..."
  if ! check_lock_state; then
    exit 10
  fi
  launch_watch_foreground
  sleep 2
  if run_test_attempt 2; then
    echo "Watch device tests passed on retry."
    exit 0
  fi
  print_clock_state_guidance
fi

echo "Watch device tests failed."
echo "Log: ${LAST_LOG_FILE}"
echo "Result bundle: ${LAST_RESULT_BUNDLE}"
exit 1
