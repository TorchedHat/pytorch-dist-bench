# pytorch-dist-bench

Microbenchmark suite for PyTorch distributed operations. Tracks performance across PyTorch releases on GPU clusters, with a focus on the collective operations that dominate tensor-parallel inference and FSDP2 training, including `torch.compile` interactions.

Designed for single-node multi-GPU systems (tested on 8×H200 NVSwitch). All benchmarks produce structured JSON for automated regression detection.

## Quick start

```bash
# Run all 9 benchmarks on 8 GPUs, write JSON to ./results/
./run_all.sh 8

# Run a single benchmark
torchrun --nproc_per_node=8 bench_collectives.py --json results/collectives.json

# Compare two runs
python compare_results.py results/baseline/ results/test/ --threshold 5
```

## Requirements

- PyTorch 2.6+ (for `fully_shard`, symmetric memory, FP8 fused ops)
- CUDA 12.x
- NCCL 2.21+
- 2+ NVIDIA GPUs (8× with NVSwitch recommended)

Some benchmarks require NVSwitch and symmetric memory support (see table below). The remainder run on any multi-GPU system.

## Benchmarks

| Benchmark | What it measures | NVSwitch required? |
|---|---|---|
| `bench_collectives` | AllReduce, AllGather, ReduceScatter at 11 message sizes (1KB–1GB) through `torch.distributed`. The nccl-tests equivalent through the ProcessGroup stack. | No |
| `bench_symm_mem_fused_ops` | BF16 fused GEMM+ReduceScatter, AllGather+GEMM, NVLS AllReduce via `torch.distributed._symmetric_memory`. The TP inference fast path. | Yes |
| `bench_fp8_fused_ops` | FP8 scaled fused ops (`_fused_all_gather_scaled_matmul`, `_fused_scaled_matmul_reduce_scatter`) vs unfused equivalents. The quantized inference path. | Yes |
| `bench_migration_path` | pynccl → `torch.distributed` → fused ops progression. Validates that migrating dispatch paths doesn't regress and that fused ops improve latency. | Yes |
| `bench_inference_tp_layer` | Full TP transformer layer (attention + MLP) with fused vs unfused collectives. Composite benchmark at real Llama-70B dimensions. | Yes |
| `bench_training_fsdp_collectives` | FSDP2-shaped AllGather/ReduceScatter, low-contention AllGather (copy engine), NVLS AllReduce. Raw collective ops at FSDP parameter shard sizes. | Yes |
| `bench_fsdp2_training` | Complete FSDP2 training step (`fully_shard()` → zero_grad → forward → backward → optimizer.step) on MLP blocks at Llama-70B dimensions. Tests FSDP2's overlap scheduling end-to-end. | No |
| `bench_moe_alltoall` | MoE expert-parallel all-to-all dispatch with balanced and skewed (Zipf) routing. Mixtral-8x7B and DeepSeek-V2 shapes. | No |
| `bench_allreduce_dispatch` | CPU dispatch overhead: pynccl vs ProcessGroupNCCL, with CUDA event timing and CUDA graph variants. | No |
| `bench_compile_distributed` | `torch.compile` (Inductor) vs eager on FSDP2 training steps and TP-style inference. Tracks whether compile helps, hurts, or breaks distributed workloads across releases. | No |

### Portable subset

5 benchmarks run on any multi-GPU system without NVSwitch or symmetric memory: `bench_collectives`, `bench_fsdp2_training`, `bench_moe_alltoall`, `bench_allreduce_dispatch`, `bench_compile_distributed`.

## JSON output

Every benchmark writes structured JSON through `bench_utils.write_json()`:

```json
{
  "benchmark": "collectives",
  "timestamp": "2026-07-14T18:30:00+00:00",
  "pytorch_version": "2.8.0a0+git1234abc",
  "pytorch_commit": "1234abc",
  "cuda_version": "12.6",
  "nccl_version": "2.25.1",
  "gpu": "NVIDIA H200",
  "gpu_count": 8,
  "dp": 8,
  "dtype": "bf16",
  "results": [
    {
      "collective": "all_reduce",
      "nelems": 536870912,
      "nbytes": 1073741824,
      "stats": {
        "p50_us": 1234.5,
        "mean_us": 1250.3,
        "p5_us": 1200.1,
        "p95_us": 1310.2,
        "min_us": 1195.0,
        "max_us": 1450.8,
        "iqr_us": 45.2,
        "iters": 200
      },
      "algo_bw_gbps": 810.5,
      "bus_bw_gbps": 709.2
    }
  ]
}
```

The `p50_us` field (median latency in microseconds) is the primary metric used for regression detection.

## Comparing results

`compare_results.py` matches JSON files by filename between two result directories, extracts `p50_us` metrics, and flags regressions:

```bash
python compare_results.py results/baseline/ results/test/
```

```
=== bench_collectives_tp8.json ===
  all_reduce  nelems=536870912      stats    1234.5 ->  1298.7  (+5.2%)  REGRESSION
  all_gather  nelems=536870912      stats    1100.2 ->  1045.1  (-5.0%)  IMPROVED
  ...

Summary: 42 metrics compared
  1 REGRESSIONS (>5.0% slower)
  1 improvements (<-5.0% faster)
  40 unchanged (within +/-5.0%)
```

Exit code is non-zero when any regression exceeds the threshold (default: 5%).

## A/B testing a PyTorch PR

`ab_test_pytorch_pr.sh` automates the full workflow: build baseline → run benchmarks → apply PR → rebuild → run benchmarks → compare.

```bash
./ab_test_pytorch_pr.sh 187642 8    # Test PR #187642 on 8 GPUs
./ab_test_pytorch_pr.sh abc1234 4   # Test a specific commit on 4 GPUs
```

Requires a PyTorch source checkout (defaults to `/workspaces/cuda-dev-env/pytorch`).

## Measurement methodology

All benchmarks share infrastructure through `bench_utils.py`:

- **`bench(fn, warmup=50, iters=200)`** — CUDA-synchronous timing with `torch.cuda.synchronize()` before each clock read. Reports p50, p5, p95, IQR. Flags runs where IQR/median exceeds 10%.
- **`reset_nccl_tuning(fn, warmup=20)`** — Barrier + warmup between configurations. NCCL's runtime tuner explores algorithms when tensor sizes change; without this reset, the first iterations at a new size use a suboptimal algorithm and inject multi-millisecond spikes.
- **`collect_metadata(name, **kwargs)`** — Captures PyTorch version, commit SHA, CUDA/NCCL versions, GPU model, and parallelism configuration.
- **Sequential execution** — `run_all.sh` runs benchmarks one at a time to prevent GPU contention from corrupting measurements.

## Project structure

```
bench_utils.py                  # Shared timing, stats, metadata, JSON output
bench_collectives.py            # Raw collective sweep (AR/AG/RS × 11 sizes)
bench_symm_mem_fused_ops.py     # BF16 fused ops (symmetric memory)
bench_fp8_fused_ops.py          # FP8 scaled fused ops
bench_migration_path.py         # pynccl → dist → fused migration
bench_inference_tp_layer.py     # Full TP layer (attention + MLP)
bench_training_fsdp_collectives.py  # FSDP2-shaped raw collectives
bench_fsdp2_training.py         # FSDP2 training step (fully_shard)
bench_moe_alltoall.py           # MoE expert-parallel all-to-all
bench_allreduce_dispatch.py     # Dispatch overhead comparison
bench_compile_distributed.py    # torch.compile vs eager (FSDP2 + TP)
run_all.sh                      # Sequential runner for all benchmarks
ab_test_pytorch_pr.sh           # A/B test harness for PyTorch PRs
compare_results.py              # JSON regression detector
```

## License

Apache-2.0
