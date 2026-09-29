#!/usr/bin/env bash
# CS-777 Chapter 9 — run the whole GDELT demo, one Terminal window per program.
#
#   "code/scripts/09.06 Run GDELT Demo.sh"
#
# Opens three macOS Terminal windows: the feeder (09.03), the Spark streaming
# query (09.04) and the Flask dashboard (09.05), then opens the dashboard in
# the browser. All three start in the folder you ran this from, so they share
# ./gdelt-input and ./gdelt-dashboard. Stop each one with Ctrl-C in its window.
#
# GDELT publishes every 15 minutes, so the charts change about that often.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# The project's virtual environment if there is one, otherwise python3 on the PATH.
PY="$ROOT/.venv/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"
WORK="$PWD"

PORT=8050  # fixed in 09.05 GDELT Dashboard.py

# A snapshot left over from an earlier run would be shown until the first new
# batch arrives, so start the dashboard from nothing.
rm -f "$WORK/gdelt-dashboard/latest.json"

# Open one Terminal window running the given command, titled with the given
# name. The command is passed to osascript as an argument rather than pasted
# into the AppleScript source, so paths with spaces need no extra escaping.
open_window() {
    local title="$1" command="$2"
    osascript - "$title" "$command" >/dev/null <<'APPLESCRIPT'
on run argv
    tell application "Terminal"
        activate
        set newTab to do script (item 2 of argv)
        set custom title of newTab to (item 1 of argv)
    end tell
end run
APPLESCRIPT
}

# Every window starts in the same folder, with the project's Spark environment
# when spark-env.sh exists.
SETUP="cd $(printf %q "$WORK")"
[ -f "$ROOT/spark-env.sh" ] && SETUP="$SETUP && source $(printf %q "$ROOT/spark-env.sh")"
script() { printf %q "$HERE/$1"; }

open_window "09.03 GDELT Feed" \
    "$SETUP && $(printf %q "$PY") $(script '09.03 GDELT Feed.py')"
open_window "09.04 GDELT Stream" \
    "$SETUP && $(printf %q "$PY") $(script '09.04 GDELT Stream.py')"
open_window "09.05 GDELT Dashboard" \
    "$SETUP && $(printf %q "$PY") $(script '09.05 GDELT Dashboard.py')"

# Give Flask a moment to start listening before the browser asks for the page.
sleep 3
open "http://127.0.0.1:$PORT"

echo "Started the GDELT demo in $WORK"
echo "  dashboard: http://127.0.0.1:$PORT"
echo "Stop each program with Ctrl-C in its Terminal window."
