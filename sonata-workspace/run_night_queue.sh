#!/usr/bin/env bash
# Autonomous night queue for apr17_morning GPU experiments.
# Order:
#   1. Wait for LiDiff eval (pgrep lidiff_eval_v2gt.py) to finish
#   2. Save LiDiff summary
#   3. Scaffold-quality sweep
#   4. N-seed ensemble
#   5. Iterative self-scaffolding
#   6. Hardware latency benchmark
#   7. Assemble summary JSON and send Telegram notification
#
# Each step runs in its own python process; failures are logged but never block.

set -u  # don't set -e — we want to keep going past failures

WORK_DIR="/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
RESULTS_DIR="${WORK_DIR}/results/apr17_morning"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"
SUMMARY="${RESULTS_DIR}/summary.json"

cd "$WORK_DIR" || { echo "cannot cd to $WORK_DIR"; exit 1; }

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

notify() {
    # send telegram; ignore errors
    ssh business "telegram-notify \"$1\"" 2>/dev/null || echo "[warn] telegram failed: $1"
}

run_step() {
    local name="$1"
    local script="$2"
    local log_file="${LOG_DIR}/${name}.log"
    log "=== START $name ==="
    local t0=$(date +%s)
    if python3 "$script" > "$log_file" 2>&1; then
        local t1=$(date +%s)
        local dt=$((t1 - t0))
        log "=== DONE  $name (${dt}s, OK) ==="
        notify "Step $name: DONE (${dt}s)"
        return 0
    else
        local t1=$(date +%s)
        local dt=$((t1 - t0))
        log "=== FAIL  $name (${dt}s) — see $log_file ==="
        notify "Step $name: FAILED (${dt}s) — check $log_file"
        return 1
    fi
}

# -------- Step 1: wait for LiDiff --------
log "Waiting for LiDiff eval to finish..."
notify "Night queue: waiting for LiDiff to finish..."
while pgrep -f lidiff_eval_v2gt.py > /dev/null 2>&1; do
    sleep 30
done
log "LiDiff process gone."

# Save LiDiff summary to results dir
LIDIFF_LOG="/home/anywherevla/lidiff_full_eval.log"
LIDIFF_JSON="/home/anywherevla/lidiff_on_v2gt_results.json"
if [ -f "$LIDIFF_JSON" ]; then
    cp "$LIDIFF_JSON" "${RESULTS_DIR}/lidiff_on_v2gt_results.json"
    log "Copied LiDiff JSON."
fi
if [ -f "$LIDIFF_LOG" ]; then
    tail -100 "$LIDIFF_LOG" > "${RESULTS_DIR}/lidiff_eval_tail.log" 2>/dev/null || true
fi
notify "LiDiff eval finished. Starting scaffold sweep."

# Short pause to let GPU cool
sleep 5

# -------- Step 2: Scaffold-quality sweep --------
run_step "scaffold_sweep" "${WORK_DIR}/run_scaffold_sweep.py"

sleep 3

# -------- Step 3: N-seed ensemble --------
run_step "n_seed_ensemble" "${WORK_DIR}/run_n_seed_ensemble.py"

sleep 3

# -------- Step 4: Iterative self-scaffolding --------
run_step "iterative_scaffolding" "${WORK_DIR}/run_iterative_scaffolding.py"

sleep 3

# -------- Step 5: Hardware latency --------
run_step "latency_benchmark" "${WORK_DIR}/run_latency_benchmark.py"

# -------- Assemble summary --------
log "Assembling summary JSON..."
python3 - <<'PY' > "${LOG_DIR}/summary_build.log" 2>&1
import json
from pathlib import Path

RESULTS = Path("/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace/results/apr17_morning")
out = {}

files = {
    "scaffold_quality_sweep": RESULTS / "scaffold_quality_sweep.json",
    "n_seed_ensemble": RESULTS / "n_seed_ensemble.json",
    "iterative_scaffolding": RESULTS / "iterative_scaffolding.json",
    "latency_benchmark": RESULTS / "latency_benchmark.json",
    "lidiff_on_v2gt": RESULTS / "lidiff_on_v2gt_results.json",
}
for key, path in files.items():
    if path.exists():
        try:
            with open(path) as f:
                out[key] = json.load(f)
        except Exception as e:
            out[key] = {"load_error": str(e)}
    else:
        out[key] = {"missing": True}

# Also a flat "headline" block
headline = {}
# scaffold sweep: extract baseline and worst
sw = out.get("scaffold_quality_sweep", {}).get("variants", {})
for name, r in sw.items():
    if isinstance(r, dict) and "pred_cd_mean" in r:
        headline[f"scaffold:{name}"] = round(r["pred_cd_mean"], 4)
# ensemble headline
ens = out.get("n_seed_ensemble", {}).get("results", {})
for mod, r in ens.items():
    if isinstance(r, dict):
        for k in ("n1_cd_mean", "n2_cd_mean", "n4_cd_mean", "n8_cd_mean"):
            if k in r:
                headline[f"ensemble:{mod}:{k}"] = round(r[k], 4)
# iterative headline
it = out.get("iterative_scaffolding", {}).get("results", {})
for variant, r in it.items():
    if isinstance(r, dict):
        for k, v in r.items():
            if k.startswith("iter") and k.endswith("_cd_mean"):
                headline[f"iterative:{variant}:{k}"] = round(v, 4)
# latency headline
lat = out.get("latency_benchmark", {})
if "configs" in lat:
    for label, r in lat["configs"].items():
        if isinstance(r, dict) and "full_mean_s" in r:
            headline[f"latency:{label}_ms"] = round(r["full_mean_s"] * 1000, 2)
            headline[f"latency:{label}_fps"] = round(r.get("full_fps", 0.0), 1)

out["_headline"] = headline

with open(RESULTS / "summary.json", "w") as f:
    json.dump(out, f, indent=2)

print("Summary keys:")
for k in out:
    v = out[k]
    if isinstance(v, dict):
        if "missing" in v:
            print(f"  {k}: MISSING")
        elif "load_error" in v:
            print(f"  {k}: LOAD ERROR {v['load_error']}")
        else:
            print(f"  {k}: OK ({len(v)} keys)")

print("\nHeadline:")
for k, v in headline.items():
    print(f"  {k} = {v}")
PY
log "Summary written to ${SUMMARY}"

# -------- Final notification --------
notify "All GPU experiments complete. Results in results/apr17_morning/. Scaffold sweep, N-seed ensemble, iterative scaffolding, hardware latency."
log "Night queue finished."
