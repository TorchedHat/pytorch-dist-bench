#!/bin/bash
# A/B test a PyTorch PR against the distributed benchmark suite.
#
# Builds PyTorch twice (with and without a PR), runs the benchmarks,
# and produces JSON results for comparison. Traces a specific PyTorch
# commit to measurable distributed performance deltas.
#
# Usage:
#   ./ab_test_pytorch_pr.sh <pr_number_or_commit> [nproc] [pytorch_dir]
#
# Examples:
#   ./ab_test_pytorch_pr.sh 187642 8                       # Test PR #187642 on 8 GPUs
#   ./ab_test_pytorch_pr.sh abc1234 4 /opt/pytorch         # Custom PyTorch source path
#   PYTORCH_DIR=/opt/pytorch ./ab_test_pytorch_pr.sh 187642 8  # Via env var
#
# Prerequisites:
#   - PyTorch source checkout (set PYTORCH_DIR or pass as 3rd arg)
#   - CUDA toolkit available

set -euo pipefail

PR_OR_COMMIT="${1:?Usage: $0 <pr_number_or_commit> [nproc] [pytorch_dir]}"
NPROC="${2:-2}"
PYTORCH_DIR="${3:-${PYTORCH_DIR:-$(python -c 'import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null || echo "")}}"

if [ -z "$PYTORCH_DIR" ] || [ ! -d "$PYTORCH_DIR/.git" ]; then
    echo "Error: PyTorch source directory not found."
    echo "Set PYTORCH_DIR, pass as 3rd argument, or install PyTorch from source."
    echo "Usage: $0 <pr_number_or_commit> [nproc] [pytorch_dir]"
    exit 1
fi
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${BENCH_DIR}/results"

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "A/B Test: PyTorch PR/commit ${PR_OR_COMMIT}"
echo "  GPUs: ${NPROC}"
echo "  PyTorch: ${PYTORCH_DIR}"
echo "  Results: ${RESULTS_DIR}"
echo "============================================================"

# --- Phase 1: Baseline (current HEAD) ---
echo ""
echo "=== Phase 1: Building baseline (current HEAD) ==="
cd "$PYTORCH_DIR"
BASELINE_SHA=$(git rev-parse --short HEAD)
echo "  Baseline: ${BASELINE_SHA}"

BASELINE_STAMP="${RESULTS_DIR}/.built_${BASELINE_SHA}"
if [ ! -f "$BASELINE_STAMP" ]; then
    echo "  Building PyTorch at ${BASELINE_SHA}..."
    python setup.py develop 2>&1 | tail -5
    touch "$BASELINE_STAMP"
else
    echo "  Already built at ${BASELINE_SHA}, skipping"
fi

echo "  Running all benchmarks..."
BASELINE_DIR="${RESULTS_DIR}/${TIMESTAMP}_baseline_${BASELINE_SHA}"
BASELINE_TXT="${RESULTS_DIR}/${TIMESTAMP}_baseline_${BASELINE_SHA}.txt"
"${BENCH_DIR}/run_all.sh" "$NPROC" --json-dir "$BASELINE_DIR" \
    2>&1 | tee "$BASELINE_TXT"

# --- Phase 2: Apply PR/commit ---
echo ""
echo "=== Phase 2: Building with PR/commit ${PR_OR_COMMIT} ==="

if [[ "$PR_OR_COMMIT" =~ ^[0-9]+$ ]]; then
    echo "  Fetching PR #${PR_OR_COMMIT}..."
    git fetch origin "pull/${PR_OR_COMMIT}/head:pr-${PR_OR_COMMIT}" 2>/dev/null || {
        echo "  Not a PR number, treating as commit SHA"
        git checkout "$PR_OR_COMMIT"
    }
    if git rev-parse "pr-${PR_OR_COMMIT}" >/dev/null 2>&1; then
        git checkout "pr-${PR_OR_COMMIT}"
    fi
else
    git checkout "$PR_OR_COMMIT"
fi

TEST_SHA=$(git rev-parse --short HEAD)
echo "  Test: ${TEST_SHA}"

echo "  Building PyTorch at ${TEST_SHA}..."
python setup.py develop 2>&1 | tail -5

echo "  Running all benchmarks..."
TEST_DIR="${RESULTS_DIR}/${TIMESTAMP}_test_${TEST_SHA}"
TEST_TXT="${RESULTS_DIR}/${TIMESTAMP}_test_${TEST_SHA}.txt"
"${BENCH_DIR}/run_all.sh" "$NPROC" --json-dir "$TEST_DIR" \
    2>&1 | tee "$TEST_TXT"

# --- Phase 3: Restore baseline ---
echo ""
echo "=== Restoring baseline ==="
git checkout "$BASELINE_SHA"
python setup.py develop 2>&1 | tail -5

# --- Phase 4: Compare ---
echo ""
echo "============================================================"
echo "Results saved:"
echo "  Baseline: ${BASELINE_DIR}/"
echo "  Test:     ${TEST_DIR}/"
echo ""
echo "  Human-readable:"
echo "    diff ${BASELINE_TXT} ${TEST_TXT}"
echo ""
echo "  JSON comparison (per-benchmark):"
echo "    for f in ${BASELINE_DIR}/*.json; do"
echo "      echo \"=== \$(basename \$f) ===\""
echo "      diff <(python -m json.tool \"\$f\") <(python -m json.tool \"${TEST_DIR}/\$(basename \$f)\")"
echo "    done"
echo "============================================================"
