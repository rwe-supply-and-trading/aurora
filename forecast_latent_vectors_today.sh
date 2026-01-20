#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# USAGE:
#   ./forecast_latent_vectors_today.sh \
#     START_TIME \
#     END_TIME \
#     SRC_REPO \
#     SRC_BRANCH \
#     DST_REPO \
#     DST_BRANCH \
#     [LAT_MIN] \
#     [LAT_MAX] \
#     [LON_MIN] \
#     [LON_MAX] \
#     [INIT_HOUR] 
# -----------------------------

# ---- clear problematic SSL / proxy env vars ----
unset CURL_CA_BUNDLE
unset REQUESTS_CA_BUNDLE
unset SSL_CERT_FILE
unset SSL_CERT_DIR

# --------------------------------------------------
# Required arguments
# --------------------------------------------------
START_TIME="$1"
END_TIME="$2"
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

echo
echo "================================================"
echo "Submitting ${START_TIME} → ${END_TIME}"
echo "================================================"
echo

echo "[CMD] python -u latent_forecast_writer.py submit-jobs $START_TIME → $END_TIME"

python -u latent_forecast_writer.py submit-jobs \
  "$START_TIME" \
  "$END_TIME" \
  --src-repo "$SRC_REPO" \
  --src-branch "$SRC_BRANCH" \
  --dst-repo "$DST_REPO" \
  --dst-branch "$DST_BRANCH" \
  --times-at-once "$TIMESTEPS_PER_JOB" \
  --aws-profile "$AWS_PROFILE" \
  --coordination-location "$COORDINATION_LOCATION" \
  "${EXTRA_ARGS[@]}"

echo
echo "FORECAST COMPLETE"
