#!/usr/bin/env bash
set -euo pipefail

START_YEAR=2016
END_YEAR=2025

SRC_REPO="rwe/era5-0p25-6h-nonprod-ohio"
SRC_BRANCH="main"
DST_STORE="s3://kafou/aurora-era5-samples.zarr"
TOKEN=""
# each job is loading
TIMES_AT_ONCE=180
# max jobs running at once


CPUS=4



echo "===================================================================="
echo "[2016] submitting"
echo "===================================================================="

python -u channelize.py submit-jobs \
    "2016-01-01T00:00:00" \
    "2016-06-30T18:00:00" \
    --src-repo "$SRC_REPO" \
    --src-branch "$SRC_BRANCH" \
    --dst-store "$DST_STORE" \
    --token "$TOKEN" \
    --times-at-once "$TIMES_AT_ONCE" \
    --cpus "$CPUS"

echo "===================================================================="
echo "[2016] DONE"
echo "===================================================================="




# for year in $(seq "$START_YEAR" "$END_YEAR"); do
#     start="${year}-01-01T00:00:00"
#     end="${year}-12-31T18:00:00"

#     echo "===================================================================="
#     echo "[${year}] submitting ${start} -> ${end}"
#     echo "===================================================================="

#     python -u channelize.py submit-jobs \
#         "$start" \
#         "$end" \
#         --src-repo "$SRC_REPO" \
#         --src-branch "$SRC_BRANCH" \
#         --dst-store "$DST_STORE" \
#         --token "$TOKEN" \
#         --times-at-once "$TIMES_AT_ONCE" \
#         --cpus "$CPUS"

#     echo "===================================================================="
#     echo "[${year}] DONE"
#     echo "===================================================================="

# done
