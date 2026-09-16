"""
Benchmark: PyTorch symmetric memory fused ops vs unfused collectives.

Demonstrates the performance chain:
  fused_matmul_reduce_scatter  vs  matmul + reduce_scatter
  fused_all_gather_matmul      vs  all_gather + matmul

These fused ops overlap communication with computation — the primary
lever for TP inference latency. They live in torch.distributed._symmetric_memory
and are what vLLM's AsyncTPPass (collective_fusion.py) pattern-matches into.

Tensor dimensions match real vLLM sequence-parallel TP patterns:
  RowParallel (down_proj):  A=[S, H/TP] x B=[H/TP, H] -> RS(dim=0) -> [S/TP, H]
  ColumnParallel (up_proj): AG(x=[S/TP, H], dim=0) -> [S, H] x W=[H, H/TP] -> [S, H/TP]

Usage:
  torchrun --nproc_per_node=2 bench_symm_mem_fused_ops.py
  torchrun --nproc_per_node=8 bench_symm_mem_fused_ops.py
  torchrun --nproc_per_node=8 bench_symm_mem_fused_ops.py --json results/symm_mem.json
"""

import argparse

import torch
import torch.distributed as dist
try:
    import torch.distributed._symmetric_memory as symm_mem
except (ImportError, ModuleNotFoundError):
    raise SystemExit(
        "bench_symm_mem_fused_ops requires torch.distributed._symmetric_memory "
        "(not available in this PyTorch build)"
    )

from bench_utils import (
    BENCH_NCCL_TIMEOUT, add_dtype_arg, bench, collect_metadata,
    reset_nccl_tuning, resolve_dtype, verify_close, write_json,
)

# Fused GEMM ops are 16-bit inference paths; NVLS fp32 is covered by
# bench_training_fsdp_collectives.
DTYPES = ("bf16", "fp16")


MODELS = {
    "Llama-8B": {"hidden": 4096, "intermediate": 14336},
    "Llama-70B": {"hidden": 8192, "intermediate": 28672},
    "Llama-405B": {"hidden": 16384, "intermediate": 53248},
}

SEQ_LENGTHS = [128, 512, 2048, 8192]



def bench_reduce_scatter(group_name, rank, world_size, seq_len, hidden, tp,
                         dtype=torch.bfloat16, warmup=50, iters=200):
    """RowParallelLinear pattern: A=[S, H/TP] x B=[H/TP, H] -> RS -> [S/TP, H]"""
    device = torch.device(f"cuda:{rank}")
    K = hidden // tp
    N = hidden

    A = torch.randn(seq_len, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    out_unfused = torch.empty(seq_len // tp, N, dtype=dtype, device=device)

    def unfused():
        C = torch.mm(A, B)
        dist.reduce_scatter_tensor(out_unfused, C, group=dist.group.WORLD)

    def fused():
        return symm_mem._fused_matmul_reduce_scatter(
            A, B, "sum", scatter_dim=0, group_name=group_name,
        )

    reset_nccl_tuning(unfused)

    unfused()
    torch.cuda.synchronize()
    ref = out_unfused.clone()
    out_fused = fused()
    torch.cuda.synchronize()
    verify_close("reduce_scatter", ref, out_fused)

    return bench(unfused, warmup=warmup, iters=iters), bench(fused, warmup=warmup, iters=iters)


def bench_all_gather(group_name, rank, world_size, seq_len, hidden, tp,
                     dtype=torch.bfloat16, warmup=50, iters=200):
    """ColumnParallelLinear pattern: AG(x=[S/TP, H]) -> [S, H] x W=[H, H/TP]"""
    device = torch.device(f"cuda:{rank}")
    shard = seq_len // tp
    K = hidden
    N = hidden // tp

    x = torch.randn(shard, K, dtype=dtype, device=device)
    W = torch.randn(K, N, dtype=dtype, device=device)
    gathered = torch.empty(seq_len, K, dtype=dtype, device=device)

    out_unfused = torch.empty(seq_len, N, dtype=dtype, device=device)

    def unfused():
        dist.all_gather_into_tensor(gathered, x, group=dist.group.WORLD)
        torch.mm(gathered, W, out=out_unfused)

    def fused():
        return symm_mem._fused_all_gather_matmul(
            x, [W], gather_dim=0, group_name=group_name,
        )

    reset_nccl_tuning(unfused)

    unfused()
    torch.cuda.synchronize()
    ref = out_unfused.clone()
    _, (out_fused,) = fused()
    torch.cuda.synchronize()
    verify_close("all_gather", ref, out_fused)

    return bench(unfused, warmup=warmup, iters=iters), bench(fused, warmup=warmup, iters=iters)


def bench_all_reduce(group_name, rank, world_size, nelems,
                     dtype=torch.bfloat16, warmup=50, iters=200):
    """Symmetric memory all-reduce (multimem/two-shot) vs dist.all_reduce."""
    device = torch.device(f"cuda:{rank}")
    buf = symm_mem.empty(nelems, dtype=dtype, device=device)
    symm_mem.rendezvous(buf, group_name)

    regular = torch.randn(nelems, dtype=dtype, device=device)

    def dist_ar():
        dist.all_reduce(regular, group=dist.group.WORLD)

    def symm_ar():
        buf.copy_(regular)
        if world_size in (4, 6, 8):
            torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", group_name)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(buf, "sum", group_name)

    reset_nccl_tuning(dist_ar)

    return bench(dist_ar, warmup=warmup, iters=iters), bench(symm_ar, warmup=warmup, iters=iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=SEQ_LENGTHS)
    add_dtype_arg(parser, DTYPES)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", metavar="PATH",
                        help="Write JSON results to PATH (rank 0 only)")
    args = parser.parse_args()

    dtype = resolve_dtype(args)

    dist.init_process_group(backend="nccl", timeout=BENCH_NCCL_TIMEOUT)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    group_name = dist.group.WORLD.group_name

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []

    if rank == 0:
        print(f"\n{'=' * 90}")
        print(f"Symmetric Memory Fused Ops Benchmark")
        print(f"  World size (TP): {world_size}")
        print(f"  Device: {torch.cuda.get_device_name(device)}")
        print(f"  Dtype: {args.dtype}")
        print(f"{'=' * 90}")

    # ---- Section 1: Fused GEMM + Reduce-Scatter ----
    if rank == 0:
        print(f"\n--- GEMM + Reduce-Scatter (RowParallelLinear pattern) ---")
        print(f"  Pattern: A=[S, H/TP] x B=[H/TP, H] -> reduce_scatter(dim=0) -> [S/TP, H]")
        print()
        hdr = f"{'model':>12} {'S':>6} {'A shape':>16} {'B shape':>14} | {'unfused':>10} {'fused':>10} {'speedup':>8} {'hiding':>8}"
        print(hdr)
        print("-" * len(hdr))

    for model_name in args.models:
        cfg = MODELS[model_name]
        if cfg["hidden"] % world_size != 0:
            continue
        for seq_len in args.seq_lengths:
            if seq_len < world_size or seq_len % world_size != 0:
                continue
            try:
                s_unfused, s_fused = bench_reduce_scatter(
                    group_name, rank, world_size, seq_len,
                    cfg["hidden"], world_size, dtype,
                    warmup=args.warmup, iters=args.iters,
                )
                if rank == 0:
                    speedup = s_unfused["p50_us"] / max(s_fused["p50_us"], 0.1)
                    hiding = max(0, (1 - s_fused["p50_us"] / s_unfused["p50_us"])) * 100
                    K = cfg["hidden"] // world_size
                    N = cfg["hidden"]
                    print(
                        f"{model_name:>12} {seq_len:>6}"
                        f" [{seq_len:>5}x{K:<5}]"
                        f" [{K:>5}x{N:<5}]"
                        f" | {s_unfused['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                        f" {speedup:>7.2f}x {hiding:>6.1f}%"
                    )
                    json_results.append({
                        "op": "reduce_scatter",
                        "model": model_name,
                        "seq_len": seq_len,
                        "A_shape": [seq_len, K],
                        "B_shape": [K, N],
                        "unfused": s_unfused,
                        "fused": s_fused,
                        "speedup": round(speedup, 3),
                        "hiding_pct": round(hiding, 1),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>12} {seq_len:>6} FAILED: {e}")

    # ---- Section 2: Fused All-Gather + GEMM ----
    if rank == 0:
        print(f"\n--- All-Gather + GEMM (ColumnParallelLinear pattern) ---")
        print(f"  Pattern: all_gather(x=[S/TP, H], dim=0) -> [S, H] x W=[H, H/TP]")
        print()
        hdr = f"{'model':>12} {'S':>6} {'x shape':>16} {'W shape':>14} | {'unfused':>10} {'fused':>10} {'speedup':>8} {'hiding':>8}"
        print(hdr)
        print("-" * len(hdr))

    for model_name in args.models:
        cfg = MODELS[model_name]
        if cfg["hidden"] % world_size != 0:
            continue
        for seq_len in args.seq_lengths:
            if seq_len < world_size or seq_len % world_size != 0:
                continue
            try:
                s_unfused, s_fused = bench_all_gather(
                    group_name, rank, world_size, seq_len,
                    cfg["hidden"], world_size, dtype,
                    warmup=args.warmup, iters=args.iters,
                )
                if rank == 0:
                    speedup = s_unfused["p50_us"] / max(s_fused["p50_us"], 0.1)
                    hiding = max(0, (1 - s_fused["p50_us"] / s_unfused["p50_us"])) * 100
                    shard = seq_len // world_size
                    K = cfg["hidden"]
                    N = cfg["hidden"] // world_size
                    print(
                        f"{model_name:>12} {seq_len:>6}"
                        f" [{shard:>5}x{K:<5}]"
                        f" [{K:>5}x{N:<5}]"
                        f" | {s_unfused['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                        f" {speedup:>7.2f}x {hiding:>6.1f}%"
                    )
                    json_results.append({
                        "op": "all_gather",
                        "model": model_name,
                        "seq_len": seq_len,
                        "x_shape": [shard, K],
                        "W_shape": [K, N],
                        "unfused": s_unfused,
                        "fused": s_fused,
                        "speedup": round(speedup, 3),
                        "hiding_pct": round(hiding, 1),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>12} {seq_len:>6} FAILED: {e}")

    # ---- Section 3: Symm-mem all-reduce vs dist.all_reduce ----
    if rank == 0:
        print(f"\n--- Symmetric Memory All-Reduce vs dist.all_reduce ---")
        print(f"  multimem_all_reduce (NVLS) vs standard NCCL all_reduce")
        print()
        hdr = f"{'nelems':>10} {'nbytes':>10} | {'dist.ar':>10} {'symm.ar':>10} {'speedup':>8}"
        print(hdr)
        print("-" * len(hdr))

    # multimem_all_reduce_ dispatches on bf16 and fp32 only.
    nvls_ok = dtype in (torch.bfloat16, torch.float32)
    if rank == 0 and not nvls_ok:
        print(f"  skipped: multimem_all_reduce_ not implemented for {dtype}")
    ar_sizes = [1024, 4096, 16384, 65536, 262144, 1048576] if nvls_ok else []
    for nelems in ar_sizes:
        try:
            s_dist, s_symm = bench_all_reduce(
                group_name, rank, world_size, nelems, dtype,
                warmup=args.warmup, iters=args.iters,
            )
            if rank == 0:
                nbytes = nelems * dtype.itemsize
                speedup = s_dist["p50_us"] / max(s_symm["p50_us"], 0.1)
                print(
                    f"{nelems:>10} {nbytes:>10}"
                    f" | {s_dist['p50_us']:>8.1f}us {s_symm['p50_us']:>8.1f}us"
                    f" {speedup:>7.2f}x"
                )
                json_results.append({
                    "op": "all_reduce",
                    "nelems": nelems,
                    "nbytes": nbytes,
                    "dist": s_dist,
                    "symm_mem": s_symm,
                    "speedup": round(speedup, 3),
                })
        except Exception as e:
            if rank == 0:
                print(f"{nelems:>10} FAILED: {e}")

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 90}")
        print("Key:")
        print("  speedup = unfused / fused  (>1x means fused is faster)")
        print("  hiding  = communication hidden behind compute (%)")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print(f"{'=' * 90}\n")

        if args.json:
            output = collect_metadata("symm_mem_fused_ops", tp=world_size, dtype=args.dtype)
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
