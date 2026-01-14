#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# USAGE:
#   ./run_channelize_yearly.sh \
#    2009 \
#    2025 \
#    rwe/era5-0p25-6h-nonprod-ohio \
#    main \ 
#    kafou/aurora-era5-samples \
#    main \
#    $ARRAYLAKE_TOKEN
# -----------------------------

START_YEAR="$1"
END_YEAR="$2"
SRC_REPO="$3"
SRC_BRANCH="$4"
DST_REPO="$5"
DST_BRANCH="$6"
TOKEN="$7"

if [[ -z "${START_YEAR:-}" || -z "${END_YEAR:-}" || -z "${TOKEN:-}" ]]; then
  echo "Usage: $0 START_YEAR END_YEAR TOKEN"
  exit 1
fi

COORDINATION_LOCATION="s3://icechunk-write-coordination"
TIMES_AT_ONCE=14
CPUS=16

SCRIPT="channelize_data.py"

LOGDIR="logs"
mkdir -p "$LOGDIR"

# -----------------------------
# YEARLY LOOP
# -----------------------------
for ((YEAR=START_YEAR; YEAR<=END_YEAR; YEAR++)); do
  YEAR_START="${YEAR}-01-01T00:00:00"
  YEAR_END="${YEAR}-12-31T18:00:00"

  echo
  echo "=============================================="
  echo "           CHANNELIZING YEAR $YEAR"
  echo "=============================================="

  LOGFILE="$LOGDIR/channelize_${YEAR}.log"

  echo "[CMD] python -u $SCRIPT submit-jobs $YEAR_START → $YEAR_END"

  python -u "$SCRIPT" submit-jobs \
    "$YEAR_START" \
    "$YEAR_END" \
    --src-repo "$SRC_REPO" \
    --src-branch "$SRC_BRANCH" \
    --dst-repo "$DST_REPO" \
    --dst-branch "$DST_BRANCH" \
    --times-at-once "$TIMES_AT_ONCE" \
    --coordination-location "$COORDINATION_LOCATION" \
    --cpus "$CPUS" \
    --token "$TOKEN" \
    2>&1 | tee "$LOGFILE"

  echo "Finished year $YEAR"
done

echo
echo "!!! ALL YEARS COMPLETE !!!"
