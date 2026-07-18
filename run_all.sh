#!/bin/bash
# Run all pytorch-dist-bench benchmarks sequentially on N GPUs.
#
# Sequential execution prevents GPU contention that corrupts measurements.
# Each benchmark writes JSON to the results directory.
#
# Usage:
#   ./run_all.sh [nproc] [--json-dir DIR]
#
# Examples:
#   ./run_all.sh 8                          # 8 GPUs, results in ./results/
#   ./run_all.sh 2 --json-dir /tmp/bench    # 2 GPUs, results in /tmp/bench/

set -uo pipefail

NPROC="${1:-8}"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
JSON_DIR="${BENCH_DIR}/results"

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json-dir) JSON_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$JSON_DIR"

BENCHMARKS=(
    bench_verify
    bench_collectives
    bench_symm_mem_fused_ops
    bench_fp8_fused_ops
    bench_migration_path
    bench_inference_tp_layer
    bench_inference_tp_vllm
    bench_training_fsdp_collectives
    bench_fsdp2_training
    bench_pipeline_parallel
    bench_moe_alltoall
    bench_allreduce_dispatch
    bench_compile_distributed
    bench_e2e
)

PASSED=()
FAILED=()

echo "============================================================"
echo "pytorch-dist-bench: running ${#BENCHMARKS[@]} benchmarks"
echo "  GPUs: ${NPROC}"
echo "  Results: ${JSON_DIR}"
echo "============================================================"

for bench_name in "${BENCHMARKS[@]}"; do
    json_path="${JSON_DIR}/${bench_name}_tp${NPROC}.json"
    echo ""
    echo "--- ${bench_name} ---"

    if torchrun --nproc_per_node="$NPROC" \
        "${BENCH_DIR}/${bench_name}.py" \
        --json "$json_path" \
        2>&1; then
        PASSED+=("$bench_name")
    else
        FAILED+=("$bench_name")
        echo "  FAILED: ${bench_name}"
    fi
done

echo ""
echo "============================================================"
echo "Results: ${#PASSED[@]}/${#BENCHMARKS[@]} passed"
for b in "${PASSED[@]}"; do echo "  OK:   $b"; done
for b in "${FAILED[@]}"; do echo "  FAIL: $b"; done
echo "JSON: ${JSON_DIR}/"
echo "============================================================"

[ ${#FAILED[@]} -eq 0 ]
