#!/usr/bin/env bash
# CS-777 Chapter 9 — stop the Kafka cluster.
#
#   code/scripts/kafka/stop.sh
#
# Removes both containers. Topics and their messages are kept, in a named
# volume, so the next start.sh finds the cluster as you left it.
#
#   --purge   delete the data too, leaving nothing behind
#
# Nothing else on the machine is touched: the container runtime keeps running,
# and no other project's containers are affected.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
COMPOSE_FILE="$HERE/docker-compose.yml"

PURGE=0
for arg in "$@"; do
    case "$arg" in
        --purge|-v) PURGE=1 ;;
        -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]:-$0}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "stop.sh: unknown option '$arg' (try --help)" >&2; exit 2 ;;
    esac
done

find_docker() {
    if command -v docker >/dev/null 2>&1; then command -v docker; return; fi
    for candidate in \
        "$HOME/.orbstack/bin/docker" \
        "/Applications/Docker.app/Contents/Resources/bin/docker" \
        "/usr/local/bin/docker" \
        "/opt/homebrew/bin/docker"
    do
        [ -x "$candidate" ] && { echo "$candidate"; return; }
    done
}

DOCKER="$(find_docker || true)"
if [ -z "$DOCKER" ] || ! "$DOCKER" info >/dev/null 2>&1; then
    echo "Docker is not running, so there is nothing to stop."
    exit 0
fi

if [ "$PURGE" -eq 1 ]; then
    echo "Stopping Kafka and deleting its data..."
    "$DOCKER" compose -f "$COMPOSE_FILE" down -v
    echo
    echo "Stopped. Nothing was kept; the next start.sh begins with an empty cluster."
else
    echo "Stopping Kafka, keeping its topics..."
    "$DOCKER" compose -f "$COMPOSE_FILE" down
    echo
    echo "Stopped. Topics were kept; run stop.sh --purge to discard them."
fi
