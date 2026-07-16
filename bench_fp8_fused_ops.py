"""
Benchmark: FP8 scaled fused ops vs unfused equivalents.

This is the production inference path for FP8-quantized models. vLLM uses
these when serving quantized Llama/Mixtral with FP8 weights:

  fused_all_gather_scaled_matmul     vs  all_gather + fp8 scaled matmul
  fused_scaled_matmul_reduce_scatter vs  fp8 scaled matmul + reduce_scatter

Compared to the bf16 fused ops (bench_symm_mem_fused_ops.py), these take
explicit scale tensors and float8_e4m3fn inputs. The fused versions overlap
communication with the scaled GEMM — same principle, different dtype path.

Tensor shapes match TP inference patterns for Llama models at FP8 precision.

Usage:
  torchrun --nproc_per_node=2 bench_fp8_fused_ops.py
  torchrun --nproc_per_node=8 bench_fp8_fused_ops.py --json results/fp8.json
"""

import argparse

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from bench_utils import bench, collect_metadata, reset_nccl_tuning, write_json


MODELS = {
    "Llama-8B":   {"hidden": 4096, "intermediate": 14336},
    "Llama-70B":  {"hidden": 8192, "intermediate": 28672},
    "Llama-405B": {"hidden": 16384, "intermediate": 53248},
}

SEQ_LENGTHS = [128, 512, 2048, 8192]



def verify_close(name, a, b, K):
    """Correctness check for FP8 paths.

    FP8 e4m3 has ~3 mantissa bits. Different kernels (cuBLAS vs fused) use
    different tiling and accumulation order, so max_diff scales as O(sqrt(K))
    where K is the reduction dimension. We set atol = sqrt(K) * 0.15 to catch
    gross errors (silent no-ops, doubled values) without false-positives.
    """
    atol = max(1.0, K ** 0.5 * 0.15)
    if not torch.allclose(a.float(), b.float(), atol=atol, rtol=0.1):
        max_diff = (a.float() - b.float()).abs().max().item()
        raise RuntimeError(
            f"Correctness check failed for {name}: max_diff={max_diff:.4f}, "
            f"atol={atol:.2f} (K={K})")


def bench_reduce_scatter(group_name, rank, world_size, seq_len, K, N,
                         warmup=50, iters=100):
    """GEMM + RS: A=[S, K] x B=[K, N] -> RS(dim=0) -> [S/TP, N] (FP8)"""
    device = torch.device(f"cuda:{rank}")
    tp = world_size

    # B created as (N, K) then transposed — matches _scaled_mm convention
    A_fp8 = torch.randn(seq_len, K, device=device).to(torch.float8_e4m3fn)
    B_fp8 = torch.randn(N, K, device=device).to(torch.float8_e4m3fn).T
    A_scale = torch.tensor(1.0, device=device)
    B_scale = torch.tensor(1.0, device=device)
    # Unfused: scaled matmul -> reduce_scatter
    out_unfused = torch.empty(seq_len // tp, N, dtype=torch.bfloat16, device=device)

    def unfused():
        C = torch._scaled_mm(A_fp8, B_fp8, scale_a=A_scale, scale_b=B_scale,
                              out_dtype=torch.bfloat16, use_fast_accum=True)
        dist.reduce_scatter_tensor(out_unfused, C, group=dist.group.WORLD)

    def fused():
        # Fresh list each call: the impl mutates output_shape in-place
        return symm_mem._fused_scaled_matmul_reduce_scatter(
            A_fp8, B_fp8, A_scale, B_scale,
            "sum", 0, 0, group_name, [seq_len, N],
            out_dtype=torch.bfloat16, use_fast_accum=True,
        )

    reset_nccl_tuning(unfused)

    # Correctness: run both, compare
    unfused()
    torch.cuda.synchronize()
    ref = out_unfused.clone()
    out_fused = fused()
    torch.cuda.synchronize()
    verify_close("fp8_reduce_scatter", ref, out_fused, K)

    return bench(unfused, warmup=warmup, iters=iters), bench(fused, warmup=warmup, iters=iters)


def bench_all_gather(group_name, rank, world_size, seq_len, K, N,
                     warmup=50, iters=100):
    """AG + GEMM: AG(x=[S/TP, K]) -> [S, K] x W=[K, N] -> [S, N] (FP8)"""
    device = torch.device(f"cuda:{rank}")
    tp = world_size
    shard = seq_len // tp

    x_fp8 = torch.randn(shard, K, device=device).to(torch.float8_e4m3fn)
    # W created as (N, K) then transposed — matches _scaled_mm convention
    W_fp8 = torch.randn(N, K, device=device).to(torch.float8_e4m3fn).T
    x_scale = torch.tensor(1.0, device=device)
    W_scale = torch.tensor(1.0, device=device)

    # Unfused: all_gather -> scaled matmul
    gathered = torch.empty(seq_len, K, dtype=torch.float8_e4m3fn, device=device)

    def unfused():
        dist.all_gather_into_tensor(gathered, x_fp8, group=dist.group.WORLD)
        return torch._scaled_mm(gathered, W_fp8, scale_a=x_scale, scale_b=W_scale,
                                 out_dtype=torch.bfloat16, use_fast_accum=True)

    def fused():
        return symm_mem._fused_all_gather_scaled_matmul(
            x_fp8, [W_fp8], x_scale, [W_scale],
            gather_dim=0, group_name=group_name,
            biases=[None], result_scales=[None],
            out_dtypes=[torch.bfloat16], use_fast_accum=[True],
        )

    reset_nccl_tuning(unfused)

    # Correctness
    ref = unfused()
    torch.cuda.synchronize()
    ref = ref.clone()
    _, (out_fused,) = fused()
    torch.cuda.synchronize()
    verify_close("fp8_all_gather", ref, out_fused, K)

    return bench(unfused, warmup=warmup, iters=iters), bench(fused, warmup=warmup, iters=iters)


def main():
    parser = argparse.ArgumentParser(
        description="FP8 scaled fused ops benchmark")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=SEQ_LENGTHS)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", metavar="PATH",
                        help="Write JSON results to PATH (rank 0 only)")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    group_name = dist.group.WORLD.group_name

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"FP8 Scaled Fused Ops Benchmark")
        print(f"  TP: {world_size}  |  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  Input dtype: float8_e4m3fn  |  Output dtype: bfloat16")
        print(f"{'=' * 95}")

    # ---- Section 1: Fused Scaled GEMM + Reduce-Scatter ----
    if rank == 0:
        print(f"\n--- FP8 GEMM + Reduce-Scatter (RowParallelLinear) ---")
        print(f"  A=[S, H/TP](fp8) x B=[H/TP, H](fp8) -> RS(dim=0) -> [S/TP, H](bf16)")
        print()
        hdr = (f"{'model':>12} {'S':>6} {'A shape':>16} {'B shape':>14}"
               f" | {'unfused':>10} {'fused':>10} {'speedup':>8} {'hiding':>8}")
        print(hdr)
        print("-" * len(hdr))

    for model_name in args.models:
        cfg = MODELS[model_name]
        for seq_len in args.seq_lengths:
            if seq_len < world_size:
                continue
            try:
                K = cfg["hidden"] // world_size
                N = cfg["hidden"]
                s_unfused, s_fused = bench_reduce_scatter(
                    group_name, rank, world_size, seq_len, K, N,
                    warmup=args.warmup, iters=args.iters,
                )
                if rank == 0:
                    speedup = s_unfused["p50_us"] / max(s_fused["p50_us"], 0.1)
                    hiding = max(0, (1 - s_fused["p50_us"] / s_unfused["p50_us"])) * 100
                    print(
                        f"{model_name:>12} {seq_len:>6}"
                        f" [{seq_len:>5}x{K:<5}]"
                        f" [{K:>5}x{N:<5}]"
                        f" | {s_unfused['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                        f" {speedup:>7.2f}x {hiding:>6.1f}%"
                    )
                    json_results.append({
                        "op": "fp8_reduce_scatter",
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

    # ---- Section 2: Fused Scaled All-Gather + GEMM ----
    if rank == 0:
        print(f"\n--- FP8 All-Gather + GEMM (ColumnParallelLinear) ---")
        print(f"  AG(x=[S/TP, H](fp8), dim=0) -> [S, H](fp8) x W=[H, H/TP](fp8) -> [S, H/TP](bf16)")
        print()
        hdr = (f"{'model':>12} {'S':>6} {'x shape':>16} {'W shape':>14}"
               f" | {'unfused':>10} {'fused':>10} {'speedup':>8} {'hiding':>8}")
        print(hdr)
        print("-" * len(hdr))

    for model_name in args.models:
        cfg = MODELS[model_name]
        for seq_len in args.seq_lengths:
            if seq_len < world_size:
                continue
            try:
                K = cfg["hidden"]
                N = cfg["hidden"] // world_size
                shard = seq_len // world_size
                s_unfused, s_fused = bench_all_gather(
                    group_name, rank, world_size, seq_len, K, N,
                    warmup=args.warmup, iters=args.iters,
                )
                if rank == 0:
                    speedup = s_unfused["p50_us"] / max(s_fused["p50_us"], 0.1)
                    hiding = max(0, (1 - s_fused["p50_us"] / s_unfused["p50_us"])) * 100
                    print(
                        f"{model_name:>12} {seq_len:>6}"
                        f" [{shard:>5}x{K:<5}]"
                        f" [{K:>5}x{N:<5}]"
                        f" | {s_unfused['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                        f" {speedup:>7.2f}x {hiding:>6.1f}%"
                    )
                    json_results.append({
                        "op": "fp8_all_gather",
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

    # ---- Section 3: FP8 vs BF16 fused comparison ----
    if rank == 0:
        print(f"\n--- FP8 vs BF16 Fused Reduce-Scatter (same shapes) ---")
        print(f"  Compares fused FP8 path vs fused BF16 path")
        print()
        hdr = (f"{'model':>12} {'S':>6}"
               f" | {'bf16_fused':>10} {'fp8_fused':>10} {'fp8_ratio':>10}")
        print(hdr)
        print("-" * len(hdr))

    for model_name in args.models:
        cfg = MODELS[model_name]
        for seq_len in args.seq_lengths:
            if seq_len < world_size:
                continue
            try:
                K = cfg["hidden"] // world_size
                N = cfg["hidden"]

                # BF16 fused path
                A_bf16 = torch.randn(seq_len, K, dtype=torch.bfloat16, device=device)
                B_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)

                def fused_bf16():
                    return symm_mem._fused_matmul_reduce_scatter(
                        A_bf16, B_bf16, "sum", scatter_dim=0, group_name=group_name,
                    )
                s_bf16 = bench(fused_bf16, warmup=args.warmup, iters=args.iters)

                # FP8 fused path
                A_fp8 = A_bf16.to(torch.float8_e4m3fn)
                B_fp8 = torch.randn(N, K, device=device).to(torch.float8_e4m3fn).T
                A_scale = torch.tensor(1.0, device=device)
                B_scale = torch.tensor(1.0, device=device)

                def fused_fp8():
                    return symm_mem._fused_scaled_matmul_reduce_scatter(
                        A_fp8, B_fp8, A_scale, B_scale,
                        "sum", 0, 0, group_name, [seq_len, N],
                        out_dtype=torch.bfloat16, use_fast_accum=True,
                    )
                s_fp8 = bench(fused_fp8, warmup=args.warmup, iters=args.iters)

                if rank == 0:
                    ratio = s_bf16["p50_us"] / max(s_fp8["p50_us"], 0.1)
                    print(
                        f"{model_name:>12} {seq_len:>6}"
                        f" | {s_bf16['p50_us']:>8.1f}us {s_fp8['p50_us']:>8.1f}us"
                        f" {ratio:>9.2f}x"
                    )
                    json_results.append({
                        "op": "fp8_vs_bf16_fused_rs",
                        "model": model_name,
                        "seq_len": seq_len,
                        "bf16_fused": s_bf16,
                        "fp8_fused": s_fp8,
                        "fp8_speedup_over_bf16": round(ratio, 3),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>12} {seq_len:>6} FAILED: {e}")

    # ---- Section 4: MLP projections (hidden x intermediate) ----
    # These are the largest GEMMs in a transformer layer and where FP8
    # fused ops have the most communication to hide.
    if rank == 0:
        print(f"\n--- FP8 MLP Projections (gate/up: H->I/TP, down: I/TP->H) ---")
        print(f"  gate/up AG: AG(x=[S/TP, H]) -> [S, H] x W=[H, I/TP]")
        print(f"  down RS:    A=[S, I/TP] x B=[I/TP, H] -> RS -> [S/TP, H]")
        print()
        hdr = (f"{'model':>12} {'S':>6} {'proj':>5} {'A shape':>16} {'B shape':>14}"
               f" | {'unfused':>10} {'fused':>10} {'speedup':>8}")
        print(hdr)
        print("-" * len(hdr))

    for model_name in args.models:
        cfg = MODELS[model_name]
        inter = cfg["intermediate"]
        hidden = cfg["hidden"]
        for seq_len in args.seq_lengths:
            if seq_len < world_size:
                continue

            # gate/up projection: AG(x=[S/TP, H]) -> [S, H] x W=[H, I/TP]
            shard = seq_len // world_size
            N_mlp = inter // world_size
            try:
                s_unfused, s_fused = bench_all_gather(
                    group_name, rank, world_size, seq_len,
                    hidden, N_mlp,
                    warmup=args.warmup, iters=args.iters,
                )
                if rank == 0:
                    speedup = s_unfused["p50_us"] / max(s_fused["p50_us"], 0.1)
                    print(
                        f"{model_name:>12} {seq_len:>6} {'gate':>5}"
                        f" [{shard:>5}x{hidden:<5}]"
                        f" [{hidden:>5}x{N_mlp:<5}]"
                        f" | {s_unfused['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                        f" {speedup:>7.2f}x"
                    )
                    json_results.append({
                        "op": "fp8_mlp_gate_ag",
                        "model": model_name,
                        "seq_len": seq_len,
                        "x_shape": [shard, hidden],
                        "W_shape": [hidden, N_mlp],
                        "unfused": s_unfused,
                        "fused": s_fused,
                        "speedup": round(speedup, 3),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>12} {seq_len:>6} {'gate':>5} FAILED: {e}")

            # down projection: A=[S, I/TP] x B=[I/TP, H] -> RS -> [S/TP, H]
            K_mlp = inter // world_size
            try:
                s_unfused, s_fused = bench_reduce_scatter(
                    group_name, rank, world_size, seq_len,
                    K_mlp, hidden,
                    warmup=args.warmup, iters=args.iters,
                )
                if rank == 0:
                    speedup = s_unfused["p50_us"] / max(s_fused["p50_us"], 0.1)
                    print(
                        f"{model_name:>12} {seq_len:>6} {'down':>5}"
                        f" [{seq_len:>5}x{K_mlp:<5}]"
                        f" [{K_mlp:>5}x{hidden:<5}]"
                        f" | {s_unfused['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                        f" {speedup:>7.2f}x"
                    )
                    json_results.append({
                        "op": "fp8_mlp_down_rs",
                        "model": model_name,
                        "seq_len": seq_len,
                        "A_shape": [seq_len, K_mlp],
                        "B_shape": [K_mlp, hidden],
                        "unfused": s_unfused,
                        "fused": s_fused,
                        "speedup": round(speedup, 3),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>12} {seq_len:>6} {'down':>5} FAILED: {e}")

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 95}")
        print("Key:")
        print("  speedup = unfused / fused  (>1x means fused is faster)")
        print("  hiding  = communication hidden behind compute (%)")
        print("  fp8_ratio = bf16_fused / fp8_fused  (>1x means FP8 is faster)")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print(f"{'=' * 95}\n")

        if args.json:
            output = collect_metadata("fp8_fused_ops", tp=world_size, dtype="fp8_e4m3fn")
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["results"] = json_results
            write_json(args.json, output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
