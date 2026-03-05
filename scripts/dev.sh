#!/bin/bash
# SightLine 本地开发一键脚本
# 用法: ./scripts/dev.sh          — 重启后端 + 重建部署到 iPhone & Watch
#       ./scripts/dev.sh server   — 只重启后端
#       ./scripts/dev.sh build    — 只重建部署到 iPhone & Watch

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
XCODEPROJ="$PROJECT_DIR/SightLine.xcodeproj"
PYTHON="${PYTHON:-}"
PORT="${PORT:-8100}"
IPHONE_NAME="${IPHONE_NAME:-iPhone}"
WATCH_NAME="${WATCH_NAME:-Watch}"
IPHONE_ID="${IPHONE_ID:-}"
WATCH_ID="${WATCH_ID:-}"
WS_BASE_URL="${WS_BASE_URL:-}"
SCHEME="SightLine"
WATCH_SCHEME="SightLineWatch"
BUNDLE_ID="com.sunflowers.SightLine"
WATCH_BUNDLE_ID="com.sunflowers.SightLine.watchkitapp"
DERIVED_DATA="$HOME/Library/Developer/Xcode/DerivedData"
SERVER_PID_FILE="/tmp/sightline-server.pid"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

resolve_python() {
    if [ -n "$PYTHON" ] && [ -x "$PYTHON" ]; then
        return 0
    fi

    local candidates=(
        "/opt/anaconda3/envs/sightline/bin/python"
        "$PROJECT_DIR/.venv/bin/python"
        "$(command -v python3 || true)"
        "$(command -v python || true)"
    )

    for candidate in "${candidates[@]}"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            PYTHON="$candidate"
            return 0
        fi
    done

    echo -e "${RED}Cannot find a usable Python interpreter.${NC}"
    return 1
}

resolve_local_ws_url() {
    if [ -n "$WS_BASE_URL" ]; then
        return 0
    fi

    local iface ip
    iface="$(route get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
    ip=""
    if [ -n "$iface" ]; then
        ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
    fi
    if [ -z "$ip" ]; then
        ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
    fi
    if [ -z "$ip" ]; then
        ip="$(ipconfig getifaddr en1 2>/dev/null || true)"
    fi
    if [ -z "$ip" ]; then
        echo -e "${RED}Cannot detect local LAN IP for real-device WebSocket routing.${NC}"
        return 1
    fi

    WS_BASE_URL="ws://${ip}:${PORT}"
}

resolve_device_ids() {
    local devices device_lines
    devices="$(xcrun xctrace list devices 2>/dev/null || true)"
    device_lines="$(printf "%s\n" "$devices" | awk '/^== Devices ==/{capture=1;next} /^== Simulators ==/{capture=0} capture')"

    if [ -z "$IPHONE_ID" ]; then
        IPHONE_ID="$(printf "%s\n" "$device_lines" | grep -F "$IPHONE_NAME" | head -1 | sed -E 's/.*\(([A-Za-z0-9-]+)\)$/\1/' || true)"
        if [ -z "$IPHONE_ID" ]; then
            IPHONE_ID="$(printf "%s\n" "$device_lines" | grep -E "iPhone.*\([0-9]+\.[0-9]+\).*\([A-Za-z0-9-]+\)$" | head -1 | sed -E 's/.*\(([A-Za-z0-9-]+)\)$/\1/' || true)"
        fi
    fi
    if [ -z "$WATCH_ID" ]; then
        WATCH_ID="$(printf "%s\n" "$device_lines" | grep -F "$WATCH_NAME" | head -1 | sed -E 's/.*\(([A-Za-z0-9-]+)\)$/\1/' || true)"
        if [ -z "$WATCH_ID" ]; then
            WATCH_ID="$(printf "%s\n" "$device_lines" | grep -E "Watch.*\([0-9]+\.[0-9]+\).*\([A-Za-z0-9-]+\)$" | head -1 | sed -E 's/.*\(([A-Za-z0-9-]+)\)$/\1/' || true)"
        fi
    fi

    if [ -z "$IPHONE_ID" ]; then
        echo -e "${RED}Cannot find iPhone device ID. Set IPHONE_ID or IPHONE_NAME.${NC}"
        return 1
    fi
    if [ -z "$WATCH_ID" ]; then
        echo -e "${RED}Cannot find Watch device ID. Set WATCH_ID or WATCH_NAME.${NC}"
        return 1
    fi
}

kill_server() {
    local pid_from_file=""
    if [ -f "$SERVER_PID_FILE" ]; then
        pid_from_file="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
    fi

    if [ -n "$pid_from_file" ] && kill -0 "$pid_from_file" 2>/dev/null; then
        echo -e "${YELLOW}[1/7] Stopping tracked server PID $pid_from_file...${NC}"
        kill -9 "$pid_from_file" 2>/dev/null || true
        rm -f "$SERVER_PID_FILE"
        sleep 0.3
    fi

    local pids
    pids=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}[1/7] Killing old server on port $PORT...${NC}"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        rm -f "$SERVER_PID_FILE"
        sleep 0.5
    else
        echo -e "${GREEN}[1/7] Port $PORT is free${NC}"
    fi
}

start_server() {
    echo -e "${GREEN}[2/7] Starting backend server...${NC}"
    cd "$PROJECT_DIR"
    nohup "$PYTHON" server.py > /tmp/sightline-server.log 2>&1 &
    local pid=$!
    echo "$pid" > "$SERVER_PID_FILE"

    # Wait for server to be ready
    for i in {1..10}; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}      Server running (PID $pid) on port $PORT${NC}"
            echo -e "${GREEN}      Log: /tmp/sightline-server.log${NC}"
            return 0
        fi
        sleep 0.5
    done

    echo -e "${RED}      Server failed to start. Check /tmp/sightline-server.log${NC}"
    rm -f "$SERVER_PID_FILE"
    tail -n 60 /tmp/sightline-server.log || true
    return 1
}

build_and_deploy() {
    echo -e "${GREEN}      iPhone: $IPHONE_ID${NC}"
    echo -e "${GREEN}      Watch:  $WATCH_ID${NC}"
    echo -e "${GREEN}      Debug WS URL: $WS_BASE_URL${NC}"

    # --- iPhone build (includes Watch via Embed Watch Content dependency) ---
    echo -e "${GREEN}[3/7] Building iPhone app...${NC}"
    xcodebuild \
        -project "$XCODEPROJ" \
        -scheme "$SCHEME" \
        -configuration Debug \
        -destination "id=$IPHONE_ID" \
        -allowProvisioningUpdates \
        build 2>&1 | tail -5

    echo -e "${GREEN}      iPhone build succeeded${NC}"

    # --- Install to iPhone ---
    echo -e "${GREEN}[4/7] Installing to iPhone...${NC}"
    local app_path
    app_path=$(find "$DERIVED_DATA" -path "*/Debug-iphoneos/SightLine.app" -maxdepth 5 2>/dev/null | head -1)

    if [ -z "$app_path" ]; then
        echo -e "${RED}      Cannot find SightLine.app in DerivedData${NC}"
        return 1
    fi

    xcrun devicectl device install app --device "$IPHONE_ID" "$app_path" 2>&1 | tail -3
    echo -e "${GREEN}      Installed to iPhone${NC}"

    # --- Watch build ---
    echo -e "${GREEN}[5/7] Building Watch app...${NC}"
    xcodebuild \
        -project "$XCODEPROJ" \
        -scheme "$WATCH_SCHEME" \
        -configuration Debug \
        -destination "id=$WATCH_ID" \
        -allowProvisioningUpdates \
        build 2>&1 | tail -5

    echo -e "${GREEN}      Watch build succeeded${NC}"

    # --- Install to Watch ---
    echo -e "${GREEN}[6/7] Installing to Watch...${NC}"
    local watch_app_path
    watch_app_path=$(find "$DERIVED_DATA" -path "*/Debug-watchos/SightLineWatch.app" -maxdepth 5 2>/dev/null | head -1)

    if [ -z "$watch_app_path" ]; then
        echo -e "${RED}      Cannot find SightLineWatch.app in DerivedData${NC}"
        return 1
    fi

    xcrun devicectl device install app --device "$WATCH_ID" "$watch_app_path" 2>&1 | tail -3
    echo -e "${GREEN}      Installed to Watch${NC}"

    # --- Launch on iPhone ---
    echo -e "${GREEN}[7/7] Launching on iPhone...${NC}"
    local launch_env
    launch_env="$(printf '{"SIGHTLINE_WS_BASE_URL":"%s"}' "$WS_BASE_URL")"
    xcrun devicectl device process launch \
        --device "$IPHONE_ID" \
        --terminate-existing \
        --environment-variables "$launch_env" \
        "$BUNDLE_ID" 2>&1 | tail -3
    echo -e "${GREEN}      Launched${NC}"

    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}  Done! iPhone & Watch apps running.${NC}"
    echo -e "${GREEN}=========================================${NC}"
}

case "${1:-all}" in
    server)
        resolve_python
        resolve_local_ws_url
        kill_server
        start_server
        ;;
    build)
        resolve_local_ws_url
        resolve_device_ids
        build_and_deploy
        ;;
    all|"")
        resolve_python
        resolve_local_ws_url
        resolve_device_ids
        kill_server
        start_server
        build_and_deploy
        ;;
    *)
        echo "Usage: $0 [server|build|all]"
        exit 1
        ;;
esac
