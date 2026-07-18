# pytorch-dist-bench

Microbenchmark suite for PyTorch distributed operations. Tracks performance across PyTorch releases on GPU clusters. Target hardware: 8×H200 NVSwitch node.

## Quick reference

```bash
./run_all.sh 8                                          # all 14 benchmarks, 8 GPUs
torchrun --nproc_per_node=8 bench_collectives.py --json results/out.json  # single benchmark
python compare_results.py results/baseline/ results/test/                 # compare two runs
./ab_test_pytorch_pr.sh 187642 8                                          # A/B test a PR
```

## Key files

- `bench_utils.py` — shared infrastructure: `bench()`, `reset_nccl_tuning()`, `collect_metadata()`, `write_json()`
- `compare_results.py` — JSON regression detector (matches files by name, compares `p50_us`)
- `run_all.sh` — sequential runner for all 14 single-node benchmarks
- `ab_test_pytorch_pr.sh` — A/B test harness for PyTorch PRs

## Skills

This repo includes four skills in `.claude/skills/`:

- **regression-diagnosis** — diagnose a reported product regression using benchmark evidence
- **run-and-compare** — run benchmarks and compare results across versions
- **benchmark-authoring** — write new benchmarks following suite conventions
- **impact-report** — generate reports attributing improvements to team contributions

## Conventions

- All benchmarks are `bench_*.py` files launched with `torchrun`
- All timing goes through `bench_utils.bench()` (CUDA-synchronous, reports p50/IQR)
- JSON output via `collect_metadata()` + `write_json()` on rank 0
- `p50_us` is the primary metric for regression detection
