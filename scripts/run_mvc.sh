#!/usr/bin/env bash
# Minimum Viable Curriculum smoke test.
#
# Runs the tiny stage_a_mvc config across 3 seeds, captures the final eval
# (probe accuracy + MIG), and prints a clear PASS/FAIL verdict per seed and
# overall. Total wall time: ~10–20 minutes on any modern GPU.
#
# Usage:   bash scripts/run_mvc.sh [--device cuda|cpu]
# Pass criteria (per seed): probe_avg > 0.50 AND mig > 0.10
# Overall pass:             at least 2 of 3 seeds pass
#
# This is the minimum viable test of "does the architecture support concept
# emergence at all?" If this fails, the full Stage A run (10k steps, 33M
# params, hours of compute) will also fail — debug the architecture instead.

set -e

DEVICE="${1:-cuda}"
CONFIG="configs/stage_a_mvc.yaml"
SEEDS=(0 1 2)

# Per-seed pass thresholds. Tuned to be modest-but-real:
#   - probe_avg=0.50 is well above chance (~0.05–0.30 across the 5 factors)
#   - MIG=0.10 is well above what an untrained model produces (~0)
PROBE_THRESH=0.50
MIG_THRESH=0.10

mkdir -p outputs/mvc_logs

declare -a results
overall_pass=0

echo "================================================================"
echo "  Minimum Viable Curriculum smoke test"
echo "  Config: ${CONFIG}    Device: ${DEVICE}    Seeds: ${SEEDS[*]}"
echo "  Pass criteria (per seed): probe_avg > ${PROBE_THRESH}, MIG > ${MIG_THRESH}"
echo "================================================================"
echo

for seed in "${SEEDS[@]}"; do
    log="outputs/mvc_logs/seed_${seed}.log"
    out_dir="outputs/mvc/seed_${seed}"
    echo "--- Seed ${seed} (logging to ${log}) ---"
    python scripts/train_stage_a.py \
        --config "${CONFIG}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        --out-dir "${out_dir}" 2>&1 | tee "${log}"

    # Pull the LAST eval log line: e.g.
    #   [step   1000] eval: {'probe': {'shape': 0.62, 'scale': 0.41, ...}, 'mig': 0.13}
    eval_line=$(grep -E "^\[step .* eval:" "${log}" | tail -1 || true)

    if [[ -z "${eval_line}" ]]; then
        echo "[seed ${seed}] FAIL: no eval log found"
        results+=("seed=${seed} probe=NA mig=NA verdict=FAIL")
        continue
    fi

    # Parse probe values + mig from the printed dict (single quotes from Python).
    probe_avg=$(python -c "
import re, sys
line = '''${eval_line}'''
m = re.search(r\"'probe': \{([^}]*)\}\", line)
if not m: print('NA'); sys.exit()
probe_str = m.group(1)
vals = [float(v) for v in re.findall(r':\s*([0-9.]+)', probe_str)]
print(f'{sum(vals)/len(vals):.3f}' if vals else 'NA')
")
    mig=$(python -c "
import re
line = '''${eval_line}'''
m = re.search(r\"'mig':\s*([0-9.]+)\", line)
print(m.group(1) if m else 'NA')
")

    pass="FAIL"
    if [[ "${probe_avg}" != "NA" && "${mig}" != "NA" ]]; then
        if (( $(echo "${probe_avg} > ${PROBE_THRESH}" | bc -l) )) && \
           (( $(echo "${mig} > ${MIG_THRESH}" | bc -l) )); then
            pass="PASS"
            overall_pass=$((overall_pass + 1))
        fi
    fi

    echo "[seed ${seed}] probe_avg=${probe_avg} mig=${mig}  →  ${pass}"
    results+=("seed=${seed} probe_avg=${probe_avg} mig=${mig} verdict=${pass}")
    echo
done

echo "================================================================"
echo "  MVC SMOKE-TEST SUMMARY"
echo "================================================================"
for r in "${results[@]}"; do echo "  ${r}"; done
echo

if (( overall_pass >= 2 )); then
    echo "  OVERALL: PASS (${overall_pass}/3 seeds met thresholds)"
    echo "  → Architecture works on the easy case. Safe to invest in full Stage A."
    exit 0
else
    echo "  OVERALL: FAIL (only ${overall_pass}/3 seeds met thresholds)"
    echo "  → Architecture or hyperparameters need debugging BEFORE full Stage A."
    echo "  Check the per-seed logs in outputs/mvc_logs/ for divergence patterns."
    exit 1
fi
