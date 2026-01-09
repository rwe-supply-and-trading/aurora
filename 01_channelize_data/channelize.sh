#!/usr/bin/env bash
set -euo pipefail

START="$1"
END="$2"
TOKEN="$3"

SRC_REPO="rwe/model-ecmwf-t0-nonprod-frankfurt"
DEST_REPO="kafou/aurora-ecmwf-samples"
DEST_BRANCH="main"
DAYS_AT_ONCE=1

MAX_RETRIES=100
ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    if [[ $ATTEMPT -gt $MAX_RETRIES ]]; then
        echo "❌ Exceeded max retries ($MAX_RETRIES). Aborting."
        exit 1
    fi

    echo "=== Attempt $ATTEMPT: running from $START → $END ==="

    TMP_OUT=$(mktemp)

    set +e
    python -u channelize.py \
        "$START" \
        "$END" \
        --src-repo "$SRC_REPO" \
        --dst-repo "$DEST_REPO" \
        --dst-branch "$DEST_BRANCH" \
        --days-at-once "$DAYS_AT_ONCE" \
        --token "$TOKEN" \
        2>&1 | tee "$TMP_OUT"
    STATUS=${PIPESTATUS[0]}
    set -e

    # SUCCESS
    if [[ $STATUS -eq 0 ]]; then
        echo "✔️ Completed successfully."
        rm -f "$TMP_OUT"
        exit 0
    fi

    # STRICT: only trust RESTART_CURSOR printed by python
    CURSOR_LINE=$(grep -E '^RESTART_CURSOR=' "$TMP_OUT" | tail -n 1 || true)
    CURSOR="${CURSOR_LINE#RESTART_CURSOR=}"

    if [[ -z "$CURSOR_LINE" ]]; then
        echo "❌ No RESTART_CURSOR found in logs."
        echo "Aborting to avoid corrupt restart."
        rm -f "$TMP_OUT"
        exit 1
    fi

    if [[ "$CURSOR" == "NONE" ]]; then
        echo "❌ RESTART_CURSOR=NONE (no commits yet). Not safe to auto-advance."
        echo "Investigate the underlying failure; rerun from the same START when ready."
        rm -f "$TMP_OUT"
        exit 1
    fi

    # Validate cursor format
    if ! [[ "$CURSOR" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "❌ Malformed RESTART_CURSOR value: '$CURSOR'"
        rm -f "$TMP_OUT"
        exit 1
    fi

    echo "Failed after committed day: $CURSOR"
    START=$(date -I -d "$CURSOR + 1 day")

    echo "Restarting from $START ..."
    rm -f "$TMP_OUT"
done
