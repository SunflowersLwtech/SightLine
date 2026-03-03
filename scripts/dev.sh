#!/bin/bash
# SightLine 本地开发一键脚本
# 用法: ./scripts/dev.sh          — 重启后端 + 重建部署到 iPhone & Watch
#       ./scripts/dev.sh server   — 只重启后端
#       ./scripts/dev.sh build    — 只重建部署到 iPhone & Watch

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
XCODEPROJ="$PROJECT_DIR/SightLine.xcodeproj"
PYTHON="/opt/anaconda3/envs/sightline/bin/python"
PORT=8100
IPHONE_ID="00008130-0014596114D8001C"   # SunFlowers的 iPhone
WATCH_ID="00008310-0018C3A80A7B601E"    # Liu's Apple Watch
SCHEME="SightLine"
WATCH_SCHEME="SightLineWatch"
BUNDLE_ID="com.sunflowers.SightLine"
WATCH_BUNDLE_ID="com.sunflowers.SightLine.watchkitapp"
DERIVED_DATA="$HOME/Library/Developer/Xcode/DerivedData"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

kill_server() {
    local pids
    pids=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}[1/7] Killing old server on port $PORT...${NC}"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    else
        echo -e "${GREEN}[1/7] Port $PORT is free${NC}"
    fi
}

start_server() {
    echo -e "${GREEN}[2/7] Starting backend server...${NC}"
    cd "$PROJECT_DIR"
    $PYTHON server.py > /tmp/sightline-server.log 2>&1 &
    local pid=$!

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
    return 1
}

build_and_deploy() {
    # --- iPhone build (includes Watch via Embed Watch Content dependency) ---
    echo -e "${GREEN}[3/7] Building iPhone app...${NC}"
    xcodebuild \
        -project "$XCODEPROJ" \
        -scheme "$SCHEME" \
        -configuration Debug \
        -destination "id=$IPHONE_ID" \
        -allowProvisioningUpdates \
        build 2>&1 | tail -5

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}iPhone build failed.${NC}"
        return 1
    fi
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

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}Watch build failed.${NC}"
        return 1
    fi
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
    xcrun devicectl device process launch --device "$IPHONE_ID" "$BUNDLE_ID" 2>&1 | tail -3
    echo -e "${GREEN}      Launched${NC}"

    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}  Done! iPhone & Watch apps running.${NC}"
    echo -e "${GREEN}=========================================${NC}"
}

case "${1:-all}" in
    server)
        kill_server
        start_server
        ;;
    build)
        build_and_deploy
        ;;
    all|"")
        kill_server
        start_server
        build_and_deploy
        ;;
    *)
        echo "Usage: $0 [server|build|all]"
        exit 1
        ;;
esac
