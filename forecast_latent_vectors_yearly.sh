#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# USAGE:
#   ./forecast_latent_vectors_yearly.sh \
#     START_YEAR \
#     END_YEAR \
#     SRC_REPO \
#     SRC_BRANCH \
#     DST_REPO \
#     DST_BRANCH \
#     [LAT_MIN] \
#     [LAT_MAX] \
#     [LON_MIN] \
#     [LON_MAX] \
#     [INIT_HOUR] \
#     [ROLLOUT_STEPS]
# -----------------------------

# ---- clear problematic SSL / proxy env vars ----
unset CURL_CA_BUNDLE
unset REQUESTS_CA_BUNDLE
unset SSL_CERT_FILE
unset SSL_CERT_DIR

# --------------------------------------------------
# Required arguments
# --------------------------------------------------
START_YEAR="$1"
END_YEAR="$2"
SRC_REPO="$3"
SRC_BRANCH="$4"
DST_REPO="$5"
DST_BRANCH="$6"

# --------------------------------------------------
# Optional arguments
# --------------------------------------------------
LAT_MIN="${7:-}"
LAT_MAX="${8:-}"
LON_MIN="${9:-}"
LON_MAX="${10:-}"
INIT_HOUR="${11:-}"
ROLLOUT_STEPS="${12:-}"

# --------------------------------------------------
# Config
# --------------------------------------------------
AWS_PROFILE="kafou"
TIMESTEPS_PER_JOB=14
COORDINATION_LOCATION="s3://icechunk-write-coordination"

# --------------------------------------------------
# Build optional CLI args
# --------------------------------------------------
EXTRA_ARGS=()

[[ -n "$LAT_MIN"   ]] && EXTRA_ARGS+=(--lat-min "$LAT_MIN")
[[ -n "$LAT_MAX"   ]] && EXTRA_ARGS+=(--lat-max "$LAT_MAX")
[[ -n "$LON_MIN"   ]] && EXTRA_ARGS+=(--lon-min "$LON_MIN")
[[ -n "$LON_MAX"   ]] && EXTRA_ARGS+=(--lon-max "$LON_MAX")
[[ -n "$INIT_HOUR" ]] && EXTRA_ARGS+=(--init-hour "$INIT_HOUR")
[[ -n "$ROLLOUT_STEPS"  ]] && EXTRA_ARGS+=(--rollout-steps "$ROLLOUT_STEPS")

# --------------------------------------------------
# Main loop: year × month
# --------------------------------------------------

for ((Y=START_YEAR; Y<=END_YEAR; Y++)); do
  for M in 01 02 03 04 05 06 07 08 09 10 11 12; do

    MONTH_START="${Y}-${M}-01T00:00:00"
    MONTH_END="$(date -u -d "${MONTH_START} +1 month" +"%Y-%m-%dT18:00:00")"

    echo "================================================"
    echo "Submitting ${MONTH_START} → ${MONTH_END}"
    echo "================================================"

    echo "[CMD] python -u latent_forecast_writer.py submit-jobs $MONTH_START → $MONTH_END"

    python latent_forecast_writer.py submit-jobs \
      "$MONTH_START" \
      "$MONTH_END" \
      --src-repo "$SRC_REPO" \
      --src-branch "$SRC_BRANCH" \
      --dest-repo "$DST_REPO" \
      --dest-branch "$DST_BRANCH" \
      --timesteps-per-job "$TIMESTEPS_PER_JOB" \
      --aws-profile "$AWS_PROFILE" \
      --coordination-location "$COORDINATION_LOCATION" \
      "${EXTRA_ARGS[@]}"

    echo
  done
done

echo
echo "FORECAST COMPLETE"