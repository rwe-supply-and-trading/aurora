#!/usr/bin/env bash
set -euo pipefail

# YYYYMMDDHHMMSS — first timestamp to include (inclusive)
START_TS="20160101000000"
# YYYYMMDDHHMMSS — last timestamp to include (inclusive)
END_TS="20251231180000"

SRC_REPO="rwe/era5-0p25-6h-nonprod-ohio"
SRC_BRANCH="main"
DST_STORE="s3://kafou/aurora-era5-samples.zarr"
TOKEN=""
# number of 6-hourly timestamps per job (inclusive of its own start)
TIMES_AT_ONCE=730

CPUS=4

# step between timestamps, in hours
STEP_HOURS=6

# validate YYYYMMDDHHMMSS
validate_ts() {
    local name=$1 val=$2
    if [[ ! "$val" =~ ^[0-9]{14}$ ]]; then
        echo "$name must be 14 digits (YYYYMMDDHHMMSS); got: $val" >&2
        exit 1
    fi
}
validate_ts START_TS "$START_TS"
validate_ts END_TS   "$END_TS"

# YYYYMMDDHHMMSS -> epoch seconds (UTC)
ts_to_epoch() {
    local t=$1
    date -u -d "${t:0:4}-${t:4:2}-${t:6:2} ${t:8:2}:${t:10:2}:${t:12:2} UTC" +%s
}

# epoch seconds -> ISO "YYYY-MM-DDTHH:MM:SS" (UTC)
epoch_to_iso() {
    date -u -d "@$1" +%Y-%m-%dT%H:%M:%S
}

start_epoch=$(ts_to_epoch "$START_TS")
end_epoch=$(ts_to_epoch "$END_TS")

if [ "$start_epoch" -gt "$end_epoch" ]; then
    echo "START_TS ($START_TS) is after END_TS ($END_TS)" >&2
    exit 1
fi

step_seconds=$(( STEP_HOURS * 3600 ))
window_span=$(( (TIMES_AT_ONCE - 1) * step_seconds ))

current_epoch=$start_epoch
while [ "$current_epoch" -le "$end_epoch" ]; do
    window_end_epoch=$(( current_epoch + window_span ))
    if [ "$window_end_epoch" -gt "$end_epoch" ]; then
        window_end_epoch=$end_epoch
    fi

    # number of timestamps in this (possibly-truncated) window
    n_times=$(( (window_end_epoch - current_epoch) / step_seconds + 1 ))

    start_iso=$(epoch_to_iso "$current_epoch")
    end_iso=$(epoch_to_iso "$window_end_epoch")

    echo "===================================================================="
    echo "submitting ${n_times} timestamps @ ${STEP_HOURS}h: ${start_iso} -> ${end_iso}"
    echo "===================================================================="

    python -u channelize.py submit-jobs \
        "$start_iso" \
        "$end_iso" \
        --src-repo "$SRC_REPO" \
        --src-branch "$SRC_BRANCH" \
        --dst-store "$DST_STORE" \
        --token "$TOKEN" \
        --times-at-once "$TIMES_AT_ONCE" \
        --cpus "$CPUS"

    echo "===================================================================="
    echo "DONE: ${start_iso} -> ${end_iso}"
    echo "===================================================================="

    # advance to the next step after window_end
    current_epoch=$(( window_end_epoch + step_seconds ))
done
