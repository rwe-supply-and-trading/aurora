#!/bin/bash

START="$1"
END="$2"

SRC_REPO="rwe/era5-0p25-6h-nonprod-ohio"
DEST_REPO="kafou/aurora-era5-samples"
DEST_BRANCH="extend-2025"
DAYS_AT_ONCE=1
TOKEN="ema_538dadd713b94095ad24386973e7b109_fbc2a0d2fae4df6553e6d6abbfb3f66130139de2707f3021e07e6dbb39890ef3"

# -------------------------------------------------------------------
# LOOP until Python exits with code 0
# -------------------------------------------------------------------
while true; do
    echo "=== Running resample from $START → $END ==="

    # Run Python in real time, tee output to a temp file
    TMP_OUT=$(mktemp)
    
    python resample_era5.py \
        "$START" \
        "$END" \
        --src-repo "$SRC_REPO" \
        --dest-repo "$DEST_REPO" \
        --dest-branch "$DEST_BRANCH" \
        --days-at-once $DAYS_AT_ONCE \
        --token "$TOKEN" \
        2>&1 | tee "$TMP_OUT"

    STATUS=${PIPESTATUS[0]}   # captures python’s exit code even though we used pipe+tee

    # SUCCESS
    if [[ $STATUS -eq 0 ]]; then
        echo "✔️ Completed successfully."
        rm "$TMP_OUT"
        exit 0
    fi

    # FAILURE: extract only the last line (the failed day)
    LAST_DAY=$(tail -n 1 "$TMP_OUT")
    echo "❌ Failed at: $LAST_DAY"

    # Update start time and retry
    START="$LAST_DAY"
    echo "🔁 Restarting from $START ..."
    
    rm "$TMP_OUT"
done
