#!/bin/bash
# Monitor script for the vllm-radiance build (cron monitor_script pattern).
# Prints the SAME string while nothing changed (suppresses the agent run),
# prints STATE:<cur> exactly when the build state transitions.
LOG=/tmp/radiance-build.log
ST=/tmp/radiance-build-state

CUR=BUILDING
if [ -f "$LOG" ] && grep -q "BUILD EXIT: 0" "$LOG"; then
    CUR=OK
elif [ -f "$LOG" ] && grep -q "BUILD EXIT: " "$LOG"; then
    CUR=FAIL
fi
PREV=$(cat "$ST" 2>/dev/null || true)
echo "$CUR" > "$ST"
[ "$CUR" = "$PREV" ] && { echo "UNCHANGED"; exit 0; }
echo "STATE:$CUR"
