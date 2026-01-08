#!/usr/bin/env bash
set -euo pipefail

# ---- clear problematic SSL / proxy env vars ----
unset CURL_CA_BUNDLE
unset REQUESTS_CA_BUNDLE
unset SSL_CERT_FILE
unset SSL_CERT_DIR

# --------------------------------------------------
# User inputs (authoritative bounds)
# --------------------------------------------------

START_TIME="2015-01-01T00:00:00"
END_TIME="2025-11-17T18:00:00"

SRC_REPO="kafou/aurora-era5-samples"
SRC_BRANCH="extend-2025"
DEST_REPO="kafou/aurora-era5-forecast-lv-6z-rollout-geo-uk"
DEST_BRANCH="main"
AWS_PROFILE="kafou"

LAT_MIN=46.5
LAT_MAX=61.5
LON_MIN=-11.5
LON_MAX=3.5
INIT_HOUR=6

ROLLOUT_STEPS=8
TIMESTEPS_PER_JOB=14
COORDINATION_LOCATION="s3://icechunk-write-coordination"

START_TS=$(date -u -d "$START_TIME" +%s)
END_TS=$(date -u -d "$END_TIME" +%s)

# --------------------------------------------------
# Year–month loop ONLY (no time math)
# --------------------------------------------------

START_Y=$(date -u -d "$START_TIME" +%Y)
START_M=$(date -u -d "$START_TIME" +%m)
END_Y=$(date -u -d "$END_TIME" +%Y)
END_M=$(date -u -d "$END_TIME" +%m)

for ((Y=START_Y; Y<=END_Y; Y++)); do
  for M in 01 02 03 04 05 06 07 08 09 10 11 12; do

    # skip months before start
    if [[ $Y -eq $START_Y && $M -lt $START_M ]]; then
      continue
    fi

    # skip months after end
    if [[ $Y -eq $END_Y && $M -gt $END_M ]]; then
      continue
    fi

    MONTH_START="${Y}-${M}-01T00:00:00"
    MONTH_END="$(date -u -d "${MONTH_START} +1 month" +"%Y-%m-%dT18:00:00")"

    WS="$MONTH_START"
    WE="$MONTH_END"

    # clamp to global bounds
    if [[ $(date -u -d "$WS" +%s) -lt $START_TS ]]; then
      WS="$START_TIME"
    fi

    if [[ $(date -u -d "$WE" +%s) -gt $END_TS ]]; then
      WE="$END_TIME"
    fi

    # skip empty windows
    if [[ $(date -u -d "$WS" +%s) -gt $(date -u -d "$WE" +%s) ]]; then
      continue
    fi

    echo "========================================"
    echo "Submitting ${WS} → ${WE}"
    echo "========================================"

    python latent_forecast_writer.py submit-jobs \
      "$WS" \
      "$WE" \
      --src-repo "$SRC_REPO" \
      --src-branch "$SRC_BRANCH" \
      --dest-repo "$DEST_REPO" \
      --dest-branch "$DEST_BRANCH" \
      --lat-min "$LAT_MIN" \
      --lat-max "$LAT_MAX" \
      --lon-min "$LON_MIN" \
      --lon-max "$LON_MAX" \
      --init-hour "$INIT_HOUR" \
      --rollout-steps "$ROLLOUT_STEPS" \
      --timesteps-per-job "$TIMESTEPS_PER_JOB" \
      --aws-profile "$AWS_PROFILE" \
      --coordination-location "$COORDINATION_LOCATION"

  done
done
