"""Shared benchmarking infrastructure for pytorch-dist-bench.

Every benchmark imports from here instead of duplicating timing,
metadata, and statistics code. This is the single place to fix
measurement methodology.
"""

import json
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist


def bench(fn, *, warmup=50, iters=200):
    """Time a CUDA-synchronous op with proper statistical reporting.

    Uses device-level synchronize before each clock read, which correctly
    measures NCCL collective completion on the local GPU. Does NOT insert
    cross-rank barriers — each rank independently measures its local view.

    Returns a stats dict with median, IQR, percentiles, and a variance flag.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1000)

    times.sort()
    n = len(times)
    q25 = times[n // 4]
    q75 = times[3 * n // 4]
    iqr = q75 - q25
    p50 = times[n // 2]

    result = {
        "p50_us": round(p50, 1),
        "mean_us": round(sum(times) / n, 1),
        "p5_us": round(times[max(0, int(n * 0.05))], 1),
        "p95_us": round(times[int(n * 0.95)], 1),
        "min_us": round(times[0], 1),
        "max_us": round(times[-1], 1),
        "iqr_us": round(iqr, 1),
        "iters": n,
    }

    if p50 > 0 and iqr / p50 > 0.10:
        result["warning"] = f"high variance: IQR/median={iqr / p50:.0%}"

    return result


def stats(times):
    """Compute stats from a list of microsecond timings.

    Same output format as bench(), for use by benchmarks that manage
    their own timing loop (e.g. bench_allreduce_dispatch.py).
    """
    times = sorted(times)
    n = len(times)
    q25 = times[n // 4]
    q75 = times[3 * n // 4]
    iqr = q75 - q25
    p50 = times[n // 2]

    result = {
        "p50_us": round(p50, 1),
        "mean_us": round(sum(times) / n, 1),
        "p5_us": round(times[max(0, int(n * 0.05))], 1),
        "p95_us": round(times[int(n * 0.95)], 1),
        "min_us": round(times[0], 1),
        "max_us": round(times[-1], 1),
        "iqr_us": round(iqr, 1),
        "iters": n,
    }

    if p50 > 0 and iqr / p50 > 0.10:
        result["warning"] = f"high variance: IQR/median={iqr / p50:.0%}"

    return result


def reset_nccl_tuning(fn, warmup=20):
    """Barrier + warmup between configs to let NCCL re-stabilize.

    NCCL's runtime tuner explores algorithms/protocols when tensor size
    changes. Without this, the first measured iterations at a new size
    may use a suboptimal algorithm, injecting multi-millisecond spikes.

    Call this once before the bench() call when switching tensor sizes.
    bench() does its own warmup for steady-state; this handles the transition.
    """
    dist.barrier()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()


def collect_metadata(benchmark_name, **kwargs):
    """Standard JSON metadata envelope for benchmark results.

    Pass parallelism degree as kwargs: tp=8, ep=8, dp=8, etc.
    """
    nccl_ver = torch.cuda.nccl.version()
    meta = {
        "benchmark": benchmark_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": torch.__version__,
        "pytorch_commit": getattr(torch.version, "git_version", "unknown"),
        "cuda_version": torch.version.cuda or "unknown",
        "nccl_version": f"{nccl_ver[0]}.{nccl_ver[1]}.{nccl_ver[2]}",
        "gpu": torch.cuda.get_device_name(),
        "gpu_count": torch.cuda.device_count(),
    }
    meta.update(kwargs)
    return meta


def write_json(path, data):
    """Write JSON results to path (call from rank 0 only)."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON results written to {path}")
