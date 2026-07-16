"""
Benchmark: Raw collective operations through torch.distributed.

The most fundamental distributed benchmark — measures AllReduce, AllGather,
and ReduceScatter at log-spaced message sizes with no GEMM, no fused ops,
and no symmetric memory. Isolates NCCL/torch.distributed performance from
compute so regressions can be attributed correctly.

Equivalent to nccl-tests (all_reduce_perf, allgather_perf, etc.) but
going through the torch.distributed ProcessGroup stack, which is what
real PyTorch code uses.

Runs on any multi-GPU system — no NVSwitch or symmetric memory required.

Usage:
  torchrun --nproc_per_node=2 bench_collectives.py
  torchrun --nproc_per_node=8 bench_collectives.py --json results/collectives.json
"""

import argparse

import torch
import torch.distributed as dist

from bench_utils import bench, collect_metadata, reset_nccl_tuning, write_json


SIZES = [
    512,
    2048,
    8192,
    32768,
    131072,
    524288,
    2097152,
    8388608,
    33554432,
    134217728,
    536870912,
]


def algo_bw(nbytes, p50_us):
    """Algorithm bandwidth in GB/s."""
    if p50_us <= 0:
        return 0.0
    return nbytes / (p50_us * 1e-6) / 1e9


def bus_bw(algo_gbps, world_size, collective):
    """Bus bandwidth — accounts for the algorithm's communication pattern."""
    n = world_size
    if collective == "all_reduce":
        factor = 2 * (n - 1) / n
    elif collective in ("all_gather", "reduce_scatter"):
        factor = (n - 1) / n
    else:
        factor = 1.0
    return algo_gbps * factor


def bench_all_reduce(device, dtype, sizes, warmup, iters):
    results = []
    for nelems in sizes:
        tensor = torch.randn(nelems, dtype=dtype, device=device)

        def fn():
            dist.all_reduce(tensor, group=dist.group.WORLD)

        reset_nccl_tuning(fn)
        s = bench(fn, warmup=warmup, iters=iters)
        results.append(("all_reduce", nelems, s))
    return results


def bench_all_gather(device, dtype, sizes, world_size, warmup, iters):
    results = []
    for nelems in sizes:
        shard_size = nelems // world_size
        if shard_size < 1:
            continue
        actual_nelems = shard_size * world_size
        inp = torch.randn(shard_size, dtype=dtype, device=device)
        out = torch.empty(actual_nelems, dtype=dtype, device=device)

        def fn():
            dist.all_gather_into_tensor(out, inp, group=dist.group.WORLD)

        reset_nccl_tuning(fn)
        s = bench(fn, warmup=warmup, iters=iters)
        results.append(("all_gather", actual_nelems, s))
    return results


def bench_reduce_scatter(device, dtype, sizes, world_size, warmup, iters):
    results = []
    for nelems in sizes:
        shard_size = nelems // world_size
        if shard_size < 1:
            continue
        actual_nelems = shard_size * world_size
        inp = torch.randn(actual_nelems, dtype=dtype, device=device)
        out = torch.empty(shard_size, dtype=dtype, device=device)

        def fn():
            dist.reduce_scatter_tensor(out, inp, group=dist.group.WORLD)

        reset_nccl_tuning(fn)
        s = bench(fn, warmup=warmup, iters=iters)
        results.append(("reduce_scatter", actual_nelems, s))
    return results


def format_bytes(nbytes):
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KB"
    return f"{nbytes} B"


def main():
    parser = argparse.ArgumentParser(
        description="Raw collective operations benchmark (torch.distributed)")
    parser.add_argument("--dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", metavar="PATH",
                        help="Write JSON results to PATH (rank 0 only)")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"Raw Collective Operations Benchmark (torch.distributed)")
        print(f"  World size: {world_size}  |  GPU: {torch.cuda.get_device_name(device)}  |  dtype: {args.dtype}")
        print(f"  No fused ops, no symmetric memory — pure NCCL through ProcessGroup")
        print(f"{'=' * 95}")

    all_results = []

    for collective_name, bench_fn in [
        ("all_reduce", bench_all_reduce),
        ("all_gather", bench_all_gather),
        ("reduce_scatter", bench_reduce_scatter),
    ]:
        if collective_name == "all_reduce":
            results = bench_fn(device, dtype, SIZES, args.warmup, args.iters)
        else:
            results = bench_fn(device, dtype, SIZES, world_size,
                               args.warmup, args.iters)

        if rank == 0:
            print(f"\n--- {collective_name} ---")
            hdr = (f"{'nelems':>12} {'nbytes':>10}"
                   f" | {'p50_us':>10} {'algo_GB/s':>10} {'bus_GB/s':>10}")
            print(hdr)
            print("-" * len(hdr))

        for coll_name, nelems, s in results:
            nbytes = nelems * dtype.itemsize
            a_bw = algo_bw(nbytes, s["p50_us"])
            b_bw = bus_bw(a_bw, world_size, coll_name)

            if rank == 0:
                print(
                    f"{nelems:>12} {format_bytes(nbytes):>10}"
                    f" | {s['p50_us']:>8.1f}us {a_bw:>9.1f} {b_bw:>9.1f}"
                )

            all_results.append({
                "collective": coll_name,
                "nelems": nelems,
                "nbytes": nbytes,
                "stats": s,
                "algo_bw_gbps": round(a_bw, 2),
                "bus_bw_gbps": round(b_bw, 2),
            })

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"  algo_GB/s = nbytes / time  (raw throughput)")
        print(f"  bus_GB/s  = algo_GB/s * correction  (link utilization)")
        print(f"    AllReduce: 2*(N-1)/N  |  AllGather/ReduceScatter: (N-1)/N")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print(f"{'=' * 95}\n")

        if args.json:
            output = collect_metadata("collectives", dp=world_size,
                                      dtype=args.dtype)
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["results"] = all_results
            write_json(args.json, output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
