import time

import kafou_arraylake as arraylake
import numpy as np
import xarray as xr
import zarr

# --------------------------------------------------
# Config
# --------------------------------------------------
SRC_REPO = "kafou/tutorial-forecast-lv"
SRC_BRANCH = "main"

DST_REPO = "kafou/tutorial-forecast-lv-training"
DST_BRANCH = "main"

LEAD_TIMES = [6, 12, 18, 24]
INIT_HOUR = 6
TIME_CHUNK = 8

print("=== CONFIG ===")
print(f"SRC_REPO      = {SRC_REPO}:{SRC_BRANCH}")
print(f"DST_REPO      = {DST_REPO}:{DST_BRANCH}")
print(f"LEAD_TIMES    = {LEAD_TIMES}")
print(f"INIT_HOUR     = {INIT_HOUR}")
print(f"TIME_CHUNK    = {TIME_CHUNK}")
print("================\n")

# --------------------------------------------------
# Open source
# --------------------------------------------------
print("[1/7] Opening source dataset...")
client = arraylake.Client()
src = client.get_repo(SRC_REPO).readonly_session(SRC_BRANCH)

ds = xr.open_zarr(
    src.store,
    zarr_format=3,
    consolidated=False,
    chunks=None,
)

print("✔ Source opened")
print("  Original sizes:")
print({k: ds.sizes[k] for k in ds.sizes})
print()

# --------------------------------------------------
# Filter
# --------------------------------------------------
print("[2/7] Applying filters...")

ds = ds.sel(lead_time=LEAD_TIMES)
print(f"  → kept lead_time={LEAD_TIMES}")

print("init_time dtype:", ds.init_time.dtype)
print("init_time size:", ds.sizes["init_time"])

if INIT_HOUR is not None:
    before = ds.sizes["init_time"]
    ds = ds.where(ds.init_time.dt.hour == INIT_HOUR, drop=True)
    after = ds.sizes["init_time"]
    print(f"  → filtered init_hour={INIT_HOUR}: {before} → {after}")

lv = ds["lv"]

n_init = ds.sizes["init_time"]
n_lead = ds.sizes["lead_time"]
n_spatial = ds.sizes["spatial_location"]
n_feat = ds.sizes["feature"]

print("✔ Filtered dataset sizes:")
print(
    dict(
        init_time=n_init,
        lead_time=n_lead,
        spatial_location=n_spatial,
        feature=n_feat,
    )
)
print()

# --------------------------------------------------
# Build synthetic time index (no datetimes)
# --------------------------------------------------
print("[3/7] Building synthetic flattened time index...")
n_time = n_init * n_lead
time_values = np.arange(n_time, dtype="int64")
print(f"✔ synthetic time dimension created: n_time = {n_time}")
print(f"✔ time dimension created: n_time = {n_time}")
print()

# --------------------------------------------------
# Prepare destination repo / branch
# --------------------------------------------------
print("[4/7] Preparing destination repository...")
try:
    repo = client.get_repo(DST_REPO)
    print("✔ Destination repo exists")
except Exception:
    print("⚠ Destination repo not found — creating")
    repo = client.create_repo(DST_REPO)

try:
    repo.lookup_branch(DST_BRANCH)
    print(f"✔ Branch {DST_BRANCH!r} exists")
except Exception:
    print(f"⚠ Branch {DST_BRANCH!r} not found — creating from main")
    repo.create_branch(DST_BRANCH, repo.lookup_branch("main"))

dst = repo.writable_session(DST_BRANCH)
print("✔ Writable session opened\n")

# --------------------------------------------------
# Write static dataset
# --------------------------------------------------
print("[5/7] Writing static dataset (coords + attrs)...")

static = xr.Dataset(
    coords={
        "time": (
            "time",
            time_values,
            {
                "standard_name": "time",
                "definition": "init_time + lead_time",
            },
        ),
        "spatial_location": (
            "spatial_location",
            np.arange(n_spatial, dtype="int64"),
            {"long_name": "flattened (lev, lat, lon) index"},
        ),
        "feature": (
            "feature",
            np.arange(n_feat, dtype="int64"),
            {"long_name": "latent feature index"},
        ),
    },
    attrs={
        **ds.attrs,
        "valid_times": 
        "source_repo": SRC_REPO,
        "source_branch": SRC_BRANCH,
        "lead_times": LEAD_TIMES,
        "init_hour": INIT_HOUR,
    },
)

static.to_zarr(
    dst.store,
    mode="w",
    zarr_format=3,
    consolidated=False,
)

root = zarr.open_group(dst.store, mode="a", zarr_format=3)
root.attrs.clear()
root.attrs.update(static.attrs)

print("✔ Static dataset written\n")

# --------------------------------------------------
# Create LV array
# --------------------------------------------------
print("[6/7] Creating empty lv array...")
if "lv" in root:
    print("⚠ Existing lv array found — deleting")
    del root["lv"]

zarr.create_array(
    dst.store,
    name="lv",
    shape=(n_time, n_spatial, n_feat),
    chunks=(TIME_CHUNK, n_spatial, n_feat),
    dtype=np.float32,
    fill_value=np.nan,
    dimension_names=("time", "spatial_location", "feature"),
    compressors=[],
)

print("✔ lv array created\n")

# --------------------------------------------------
# Stream write
# --------------------------------------------------
print("[7/7] Streaming lv data...")
t = 0
buffer = []
last_report = time.time()

for i_init in range(n_init):
    for i_lead in range(n_lead):
        slab = lv.isel(init_time=i_init, lead_time=i_lead).values
        buffer.append(slab)

        if len(buffer) == TIME_CHUNK:
            arr = np.stack(buffer, axis=0)
            root["lv"][t : t + TIME_CHUNK, :, :] = arr
            t += TIME_CHUNK
            buffer.clear()

            if time.time() - last_report > 30:
                print(f"  wrote time[0:{t}] / {n_time}")
                last_report = time.time()

# Flush remainder
if buffer:
    arr = np.stack(buffer, axis=0)
    root["lv"][t : t + len(buffer), :, :] = arr
    t += len(buffer)

# print dataset
ds = xr.open_zarr(
    dst.store,
    zarr_format=3,
    consolidated=False,
    chunks=None,
)

print(ds)
print(f"✔ Streaming complete (wrote {t} / {n_time} time steps)\n")

# --------------------------------------------------
# Commit
# --------------------------------------------------
print("Committing...")
cid = dst.commit("forecast → inference reshape (single-node)")
print("✔ Commit complete:", cid)
