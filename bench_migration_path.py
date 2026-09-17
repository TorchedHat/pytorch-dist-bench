"""
Benchmark: The pynccl deprecation migration path.

Shows three paths for the same TP collective pattern, answering:
  1. Does migrating from pynccl to torch.distributed regress?  (No)
  2. Does the migration unlock fused ops?                       (Yes)
  3. How much do fused ops help?                                (Measured)

Paths compared:
  A) pynccl:    matmul -> pynccl.reduce_scatter     (sequential)
  B) dist:      matmul -> dist.reduce_scatter_tensor (sequential)
  C) fused:     fused_matmul_reduce_scatter          (overlapped)

Path C is only available through torch.distributed._symmetric_memory.
pynccl cannot access it. This is the performance unlock from convergence.

Usage:
  torchrun --nproc_per_node=2 bench_migration_path.py
  torchrun --nproc_per_node=8 bench_migration_path.py --json results/migration.json
"""

import argparse

import torch
import torch.distributed as dist
try:
    import torch.distributed._symmetric_memory as symm_mem
except (ImportError, ModuleNotFoundError):
    raise SystemExit(
        "bench_migration_path requires torch.distributed._symmetric_memory "
        "(not available in this PyTorch build)"
    )

from bench_utils import (
    BENCH_NCCL_TIMEOUT, bench, collect_metadata, prepare_device,
    reset_nccl_tuning, verify_close, write_json,
)


CONFIGS = [
    # (seq_len, hidden, label)
    (128,  8192, "decode-batch"),
    (512,  8192, "small-prefill"),
    (2048, 8192, "med-prefill"),
    (8192, 8192, "large-prefill"),
    (16384, 8192, "xl-prefill"),
    (32768, 8192, "xxl-prefill"),
]



def try_import_pynccl():
    try:
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        return PyNcclCommunicator
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pynccl -> torch.distributed -> fused ops migration")
    parser.add_argument("--json", metavar="PATH",
                        help="Write JSON results to PATH (rank 0 only)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", timeout=BENCH_NCCL_TIMEOUT)
    rank = dist.get_rank()
    tp = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    prepare_device(device)
    dtype = torch.bfloat16

    group_name = dist.group.WORLD.group_name

    PyNcclCommunicator = try_import_pynccl()
    pynccl_comm = None
    if PyNcclCommunicator is not None:
        try:
            gloo_group = dist.new_group(backend="gloo")
            comm = PyNcclCommunicator(group=gloo_group, device=device)
            if comm.available and not comm.disabled:
                pynccl_comm = comm
            elif rank == 0:
                print(f"  pynccl loaded but disabled (set VLLM_NCCL_SO_PATH)")
        except Exception as e:
            if rank == 0:
                print(f"  pynccl init failed: {e}")

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []
    stream = torch.cuda.current_stream()

    if rank == 0:
        print(f"\n{'=' * 88}")
        print(f"Migration Path Benchmark: pynccl -> torch.distributed -> fused ops")
        print(f"  Model: Llama-70B  |  TP: {tp}  |  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  pynccl available: {pynccl_comm is not None}")
        print(f"{'=' * 88}")
        print()
        print(f"  RowParallelLinear pattern: A=[S, H/TP] x B=[H/TP, H] -> RS(dim=0) -> [S/TP, H]")
        print()

        cols = f"{'workload':>14} {'S':>6}"
        if pynccl_comm is not None:
            cols += f" | {'pynccl':>10}"
        cols += f" | {'dist.RS':>10} {'fused':>10} {'speedup':>8} {'hiding':>8}"
        if pynccl_comm is not None:
            cols += f" | {'A->B regr':>10}"
        print(cols)
        print("-" * len(cols))

    for seq_len, hidden, label in CONFIGS:
        if seq_len < tp or seq_len % tp != 0 or hidden % tp != 0:
            continue

        K = hidden // tp
        N = hidden
        A = torch.randn(seq_len, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        out_pynccl = torch.empty(seq_len // tp, N, dtype=dtype, device=device)
        out_dist = torch.empty(seq_len // tp, N, dtype=dtype, device=device)

        s_pynccl = None
        if pynccl_comm is not None:
            def path_pynccl():
                C = torch.mm(A, B)
                pynccl_comm.reduce_scatter(out_pynccl, C, stream=stream)
            s_pynccl = bench(path_pynccl, warmup=args.warmup, iters=args.iters)

        def path_dist():
            C = torch.mm(A, B)
            dist.reduce_scatter_tensor(out_dist, C, group=dist.group.WORLD)
        s_dist = bench(path_dist, warmup=args.warmup, iters=args.iters)

        def path_fused():
            return symm_mem._fused_matmul_reduce_scatter(
                A, B, "sum", scatter_dim=0, group_name=group_name,
            )

        # Correctness check: ensure dist and fused produce similar results
        path_dist()
        torch.cuda.synchronize()
        ref = out_dist.clone()
        fused_out = path_fused()
        torch.cuda.synchronize()
        verify_close(f"RS S={seq_len}", ref, fused_out)

        s_fused = bench(path_fused, warmup=args.warmup, iters=args.iters)

        if rank == 0:
            speedup = s_dist["p50_us"] / max(s_fused["p50_us"], 0.1)
            hiding = max(0, (1 - s_fused["p50_us"] / s_dist["p50_us"])) * 100

            line = f"{label:>14} {seq_len:>6}"
            if s_pynccl is not None:
                line += f" | {s_pynccl['p50_us']:>8.1f}us"
            line += (
                f" | {s_dist['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                f" {speedup:>7.2f}x {hiding:>6.1f}%"
            )

            regression = None
            if s_pynccl is not None:
                regression = (s_dist["p50_us"] - s_pynccl["p50_us"]) / s_pynccl["p50_us"] * 100
                sign = "+" if regression > 0 else ""
                line += f" | {sign}{regression:>8.1f}%"
            print(line)

            json_results.append({
                "op": "reduce_scatter",
                "seq_len": seq_len,
                "label": label,
                "pynccl": s_pynccl,
                "dist": s_dist,
                "fused": s_fused,
                "speedup": round(speedup, 3),
                "hiding_pct": round(hiding, 1),
                "migration_regression_pct": round(regression, 2) if regression is not None else None,
            })

    # AG+GEMM direction
    if rank == 0:
        print()
        print(f"  ColumnParallelLinear pattern: AG(x=[S/TP, H], dim=0) -> [S, H] x W=[H, H/TP]")
        print()
        cols = f"{'workload':>14} {'S':>6} | {'dist.AG':>10} {'fused':>10} {'speedup':>8} {'hiding':>8}"
        print(cols)
        print("-" * len(cols))

    for seq_len, hidden, label in CONFIGS:
        if seq_len < tp or seq_len % tp != 0 or hidden % tp != 0:
            continue

        shard = seq_len // tp
        K = hidden
        N = hidden // tp
        x = torch.randn(shard, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        gathered = torch.empty(seq_len, K, dtype=dtype, device=device)
        out_dist_ag = torch.empty(seq_len, N, dtype=dtype, device=device)

        def path_dist_ag():
            dist.all_gather_into_tensor(gathered, x, group=dist.group.WORLD)
            torch.mm(gathered, W, out=out_dist_ag)

        def path_fused_ag():
            return symm_mem._fused_all_gather_matmul(
                x, [W], gather_dim=0, group_name=group_name,
            )

        path_dist_ag()
        torch.cuda.synchronize()
        ref = out_dist_ag.clone()
        _, (fused_out,) = path_fused_ag()
        torch.cuda.synchronize()
        verify_close(f"AG S={seq_len}", ref, fused_out)

        s_dist = bench(path_dist_ag, warmup=args.warmup, iters=args.iters)
        s_fused = bench(path_fused_ag, warmup=args.warmup, iters=args.iters)

        if rank == 0:
            speedup = s_dist["p50_us"] / max(s_fused["p50_us"], 0.1)
            hiding = max(0, (1 - s_fused["p50_us"] / s_dist["p50_us"])) * 100
            print(
                f"{label:>14} {seq_len:>6}"
                f" | {s_dist['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                f" {speedup:>7.2f}x {hiding:>6.1f}%"
            )
            json_results.append({
                "op": "all_gather",
                "seq_len": seq_len,
                "label": label,
                "dist": s_dist,
                "fused": s_fused,
                "speedup": round(speedup, 3),
                "hiding_pct": round(hiding, 1),
            })

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 88}")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print(f"{'=' * 88}\n")

        if args.json:
            output = collect_metadata("migration_path", tp=tp, dtype="bf16")
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["pynccl_available"] = pynccl_comm is not None
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    if pynccl_comm is not None:
        pynccl_comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
