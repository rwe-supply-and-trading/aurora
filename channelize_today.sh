#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# USAGE:
#   ./run_channelize_today.sh \
#     2025-01-01T00:00:00 \
#     2025-12-31T18:00:00 \ 
#     rwe/era5-0p25-6h-nonprod-ohio \
#     main \ 
#     kafou/aurora-era5-samples \
#     main \
#     $ARRAYLAKE_TOKEN 
# -----------------------------

START_TIME="$1"
END_TIME="$2"
SRC_REPO="$3"
SRC_BRANCH="$4"
DST_REPO="$5"
DST_BRANCH="$6"
TOKEN="$7"

if [[ -z "${START_TIME:-}" || -z "${END_TIME:-}" || -z "${TOKEN:-}" ]]; then
  echo "Usage: $0 START_TIME END_TIME SRC_REPO SRC_BRANCH DST_REPO DST_BRANCH TOKEN"
  exit 1
fi

# -----------------------------
# Config
# -----------------------------
COORDINATION_LOCATION="s3://icechunk-write-coordination"
TIMES_AT_ONCE=14
CPUS=16

SCRIPT="channelize_data.py"

LOGDIR="logs"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/channelize_${START_TIME}_to_${END_TIME}.log"
LOGFILE="${LOGFILE//:/_}"   # sanitize colons for filesystems

# -----------------------------
# Run
# -----------------------------
echo
echo "=============================================="
echo "   CHANNELIZING RANGE"
echo "       START = $START_TIME"
echo "       END   = $END_TIME"
echo "=============================================="
echo

echo "[CMD] python -u $SCRIPT submit-jobs $START_TIME → $END_TIME"

python -u "$SCRIPT" submit-jobs \
  "$START_TIME" \
  "$END_TIME" \
  --src-repo "$SRC_REPO" \
  --src-branch "$SRC_BRANCH" \
  --dst-repo "$DST_REPO" \
  --dst-branch "$DST_BRANCH" \
  --times-at-once "$TIMES_AT_ONCE" \
  --coordination-location "$COORDINATION_LOCATION" \
  --cpus "$CPUS" \
  --token "$TOKEN" \
  2>&1 | tee "$LOGFILE"

echo
echo "CHANNELIZATION COMPLETE"
echo "Log: $LOGFILE"
