#!/usr/bin/env bash
# CS-777 Chapter 9 — start the Kafka cluster.
#
#   code/scripts/kafka/start.sh
#
# Brings up one Kafka broker in KRaft mode and a browser view of it, waits until
# the broker actually answers, and prints what to run next. Safe to run twice:
# if the cluster is already up it just reports.
#
#   --fresh   discard any existing topics first, so the cluster starts empty
#
# Topics survive an ordinary stop, because the broker's log lives in a named
# volume. Use --fresh, or `stop.sh --purge`, when you want a clean slate.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
COMPOSE_FILE="$HERE/docker-compose.yml"
UI_URL="http://localhost:8080"
BOOTSTRAP="localhost:9092"

FRESH=0
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=1 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]:-$0}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "start.sh: unknown option '$arg' (try --help)" >&2; exit 2 ;;
    esac
done

# --- Find Docker -----------------------------------------------------------
# OrbStack and Docker Desktop both install a `docker` that may not be on the
# PATH of a non-login shell, so look in the usual places before giving up.
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
if [ -z "$DOCKER" ]; then
    cat >&2 <<'MSG'
No `docker` command found.

Chapter 9's Kafka example needs a container runtime. OrbStack is the lightest on
macOS:

    brew install --cask orbstack && orb start

Docker Desktop works just as well if you already have it.
MSG
    exit 1
fi

if ! "$DOCKER" info >/dev/null 2>&1; then
    cat >&2 <<'MSG'
Docker is installed but the engine is not running.

    orb start                 # OrbStack
    open -a Docker            # Docker Desktop

Then run this script again.
MSG
    exit 1
fi

# --- Bring it up -----------------------------------------------------------
if [ "$FRESH" -eq 1 ]; then
    echo "Discarding any existing Kafka data..."
    "$DOCKER" compose -f "$COMPOSE_FILE" down -v >/dev/null 2>&1 || true
fi

echo "Starting Kafka (KRaft mode, one broker) and Kafka UI..."
echo "The first run pulls about 700 MB of images; later runs take a few seconds."
"$DOCKER" compose -f "$COMPOSE_FILE" up -d

# The Compose file makes Kafka UI wait for the broker's health check, so by here
# the broker is up. Confirm it by asking it something, rather than trusting the
# exit status of `up`.
echo -n "Waiting for the broker to answer"
for _ in $(seq 1 30); do
    if "$DOCKER" exec cs777-kafka /opt/kafka/bin/kafka-topics.sh \
            --bootstrap-server "$BOOTSTRAP" --list >/dev/null 2>&1; then
        echo " ok"
        break
    fi
    echo -n "."
    sleep 2
done

if ! "$DOCKER" exec cs777-kafka /opt/kafka/bin/kafka-topics.sh \
        --bootstrap-server "$BOOTSTRAP" --list >/dev/null 2>&1; then
    echo
    echo "The broker did not come up. Its log:" >&2
    "$DOCKER" compose -f "$COMPOSE_FILE" logs --tail 40 kafka >&2
    exit 1
fi

# --- Report ----------------------------------------------------------------
echo
echo "Kafka is up."
echo "  brokers:   $BOOTSTRAP"
echo "  Kafka UI:  $UI_URL"

TOPICS="$("$DOCKER" exec cs777-kafka /opt/kafka/bin/kafka-topics.sh \
              --bootstrap-server "$BOOTSTRAP" --list 2>/dev/null | grep -v '^__' || true)"
if [ -n "$TOPICS" ]; then
    echo "  topics:    $(echo "$TOPICS" | tr '\n' ' ')"
    echo
    echo "These topics survived the last stop. Start again with --fresh for an empty cluster."
else
    echo "  topics:    none yet; the producer creates weatherdata on its first send"
fi

cat <<'NEXT'

Next, in two more terminals, from the repository root:

    source spark-env.sh
    .venv/bin/python3 "code/scripts/09.01 Weather Producer.py"
    .venv/bin/python3 "code/scripts/09.01 Kafka Weather Stream.py"

Stop the cluster with code/scripts/kafka/stop.sh
NEXT
