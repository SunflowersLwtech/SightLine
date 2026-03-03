# UI/UX Audit & Fix Plan (2026-02-22)

## Scope
- Source report: `dev/docs/Debug_Video_Display_Feasibility_Report.md`
- Goal: audit listed UI/UX gaps, implement fixes, run compatibility checks/tests, rebuild artifacts.

## Checklist
- [x] Audit current iOS/frontend implementation against report findings (F-1 ~ F-7 + debug UI gaps)
- [x] Implement downstream message compatibility fixes (`face_library_reloaded`, `face_library_cleared`, `error`)
- [x] Implement `go_away` reconnect behavior and `session_resumption` handle persistence
- [x] Implement direct force-LOD gesture pipeline (`force_lod_1/2/3`) end-to-end
- [x] Implement/verify debug UI improvements needed for developer observability (Video/Network debug where feasible in this pass)
- [x] Run focused tests (Swift unit tests + Python tests if touched)
- [x] Build/rebuild app targets and verify no regressions
- [x] Update operation log with concrete changes and validation evidence

## Review Notes
- Python regression:
  - `/opt/anaconda3/envs/sightline/bin/python -m pytest tests/test_server_integration.py -q`
  - Result: `18 passed`.
- iOS unit tests:
  - `xcodebuild -project SightLine.xcodeproj -scheme SightLine -destination 'platform=iOS Simulator,id=84112A5A-3B00-4A82-B690-CD8EF0631524' -only-testing:SightLineTests test`
  - Result: `TEST SUCCEEDED`, `82 tests passed`.
- iOS rebuild:
  - `xcodebuild -project SightLine.xcodeproj -scheme SightLine -destination 'platform=iOS Simulator,id=84112A5A-3B00-4A82-B690-CD8EF0631524' build`
  - Result: `BUILD SUCCEEDED`.

---

# Repetition Speech Bugfix Plan (2026-02-23)

## Scope
- Bug: AI repeatedly broadcasts direction/scene/heart-rate context and causes runaway speech.
- Goal: keep proactive safety output, but suppress repetitive non-actionable narration.

## Checklist
- [x] Add backend telemetry/context deduplication guard (only inject when change is meaningful).
- [x] Add context-only instructions for telemetry updates to prevent raw sensor echoing.
- [x] Add vision/OCR summary repeat suppression to avoid repeated scene narration loops.
- [x] Add downstream agent transcript repeat suppression before forwarding to iOS.
- [x] Add/adjust Python tests for telemetry parser + repeat suppression behavior.
- [x] Run local backend tests (pytest) and iOS tests/build.
- [x] Deploy backend to Cloud Run ("deploy to Claude" path in this repo).
- [x] Rebuild iPhone + iWatch targets for real device.
- [x] Update operation log with concrete commands and results.

## Review Notes
- Backend tests:
  - `/opt/anaconda3/envs/sightline/bin/python -m pytest tests/test_server_integration.py tests/test_telemetry_parser.py -q`
  - Result: `35 passed`.
  - `/opt/anaconda3/envs/sightline/bin/python -m pytest tests -q`
  - Result: `310 passed`.
- Cloud deploy:
  - `gcloud builds submit --config cloudbuild.yaml --project sightline-hackathon .`
  - Build ID: `f217f48c-ff32-4587-9d4d-4af6eabf0d99`, status `SUCCESS`.
  - Cloud Run revision: `sightline-backend-00025-5jz`, traffic `100%`.
  - Health check: `curl https://sightline-backend-kp47ssyf4q-uc.a.run.app/health` -> `status=ok`.
- iPhone real device:
  - `xcodebuild -project SightLine.xcodeproj -scheme SightLine -destination 'platform=iOS,id=00008130-0014596114D8001C' -allowProvisioningUpdates build`
  - Result: `BUILD SUCCEEDED`.
  - `xcodebuild -project SightLine.xcodeproj -scheme SightLine -destination 'platform=iOS,id=00008130-0014596114D8001C' -only-testing:SightLineTests -allowProvisioningUpdates test`
  - Result: `TEST SUCCEEDED` (`82 tests passed`).
  - Note: full suite with `SightLineUITests` includes an existing audit failure (`Potentially inaccessible text`).
- Apple Watch real device:
  - `xcodebuild -project SightLine.xcodeproj -scheme SightLineWatch -destination 'platform=watchOS,id=00008310-0018C3A80A7B601E' -allowProvisioningUpdates build`
  - Result: `BUILD SUCCEEDED`.
  - `SIGHTLINE_WATCH_DESTINATION_ID=00008310-0018C3A80A7B601E ./scripts/run_watch_device_tests.sh`
  - Result: `TEST SUCCEEDED` (`5 tests passed`).
