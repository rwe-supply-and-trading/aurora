#!/usr/bin/env bash
set -euo pipefail

START="$1"
END="$2"
TOKEN=$3


SRC_REPO="rwe/model-ecmwf-t0-nonprod-frankfurt"
DEST_REPO="kafou/aurora-ecmwf-samples"
DEST_BRANCH="testing"
DAYS_AT_ONCE=14

MAX_RETRIES=100
ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    if [[ $ATTEMPT -gt $MAX_RETRIES ]]; then
        echo "Exceeded max retries ($MAX_RETRIES). Aborting."
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
        rm "$TMP_OUT"
        exit 0
    fi

    # Extract last successfully written day (YYYY-MM-DD)
    LAST_DAY=$(grep -Eo '[0-9]{4}-[0-9]{2}-[0-9]{2}' "$TMP_OUT" | tail -n 1)

    if [[ -z "$LAST_DAY" ]]; then
        echo "Could not determine last successful day from logs."
        echo "Aborting to avoid corrupt restart."
        exit 1
    fi

    echo "Failed after day: $LAST_DAY"

    # Advance to next day
    START=$(date -I -d "$LAST_DAY + 1 day")

    echo "Restarting from $START ..."
    rm "$TMP_OUT"
done
