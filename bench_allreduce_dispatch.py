"""
Benchmark: pynccl vs ProcessGroupNCCL all-reduce dispatch overhead.

Measures the CPU-side + CUDA-side cost of the two dispatch paths:
  Path A (pynccl):  Python -> ctypes -> ncclAllReduce
  Path B (dist):    Python -> torch.distributed.all_reduce -> ProcessGroupNCCL

Includes CUDA event timing for accurate GPU-side measurement at small
tensor sizes where torch.cuda.synchronize() overhead is significant.

Usage:
  torchrun --nproc_per_node=2 bench_allreduce_dispatch.py
  torchrun --nproc_per_node=8 bench_allreduce_dispatch.py --json results/dispatch.json
"""

import argparse
import time

import torch
import torch.distributed as dist

from bench_utils import collect_metadata, stats, write_json


SIZES = [
    512,        # 1 KB (bf16)
    2048,       # 4 KB
    8192,       # 16 KB — Llama-70B single token
    16384,      # 32 KB — Llama-405B single token
    65536,      # 128 KB — small batch decode
    262144,     # 512 KB — medium batch
    1048576,    # 2 MB — large batch / prefill
    4194304,    # 8 MB — large prefill
]



def bench_pynccl(comm, tensor, warmup=50, iters=200):
    """Benchmark the pynccl direct path.

    NOTE: CUDA event timing is NOT included for pynccl. CUDA events record
    on the user stream, but pynccl dispatches to NCCL's internal streams.
    Events would not bracket the actual NCCL work — the same pitfall as
    stream.synchronize() vs torch.cuda.synchronize(). Wall-clock with
    device-level sync is the only correct GPU timing for pynccl.
    """
    stream = torch.cuda.current_stream()

    for _ in range(warmup):
        comm.all_reduce(tensor, stream=stream)
    torch.cuda.synchronize()

    cpu_times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        comm.all_reduce(tensor, stream=stream)
        cpu_times.append((time.perf_counter_ns() - t0) / 1000)
    torch.cuda.synchronize()

    wall_times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        comm.all_reduce(tensor, stream=stream)
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter_ns() - t0) / 1000)

    return {
        "cpu": stats(cpu_times),
        "gpu_wall": stats(wall_times),
    }


def bench_dist(group, tensor, warmup=50, iters=200):
    """Benchmark the torch.distributed path."""
    for _ in range(warmup):
        dist.all_reduce(tensor, group=group)
    torch.cuda.synchronize()

    cpu_times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        dist.all_reduce(tensor, group=group)
        cpu_times.append((time.perf_counter_ns() - t0) / 1000)
    torch.cuda.synchronize()

    wall_times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        dist.all_reduce(tensor, group=group)
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter_ns() - t0) / 1000)

    event_times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_ev.record()
        dist.all_reduce(tensor, group=group)
        end_ev.record()
        torch.cuda.synchronize()
        event_times.append(start_ev.elapsed_time(end_ev) * 1000)

    return {
        "cpu": stats(cpu_times),
        "gpu_wall": stats(wall_times),
        "gpu_event": stats(event_times),
    }


def bench_cuda_graph(fn_setup, warmup=5, replay=200):
    """Benchmark collective inside CUDA graph capture + replay.

    Returns both throughput (batch replay) and latency (per-replay sync) modes.
    fn_setup returns (graph, stream) after capture.
    """
    graph, stream = fn_setup()

    # Throughput mode: replay N times, one sync at end
    t0 = time.perf_counter_ns()
    for _ in range(replay):
        graph.replay()
    stream.synchronize()
    throughput_us = (time.perf_counter_ns() - t0) / replay / 1000

    # Latency mode: sync after each replay
    latency_times = []
    for _ in range(replay):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        graph.replay()
        torch.cuda.synchronize()
        latency_times.append((time.perf_counter_ns() - t0) / 1000)

    return {
        "graph_throughput_us": round(throughput_us, 2),
        "graph_latency": stats(latency_times),
    }


def setup_dist_graph(group, tensor):
    """Capture dist.all_reduce in a CUDA graph."""
    stream = torch.cuda.current_stream()
    for _ in range(5):
        dist.all_reduce(tensor, group=group)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        dist.all_reduce(tensor, group=group)
    stream.synchronize()
    return graph, stream


def setup_pynccl_graph(comm, tensor):
    """Capture pynccl all_reduce in a CUDA graph."""
    stream = torch.cuda.current_stream()
    for _ in range(5):
        comm.all_reduce(tensor, stream=stream)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        comm.all_reduce(tensor, stream=stream)
    stream.synchronize()
    return graph, stream



def main():
    parser = argparse.ArgumentParser(
        description="All-reduce dispatch overhead benchmark")
    parser.add_argument("--json", metavar="PATH",
                        help="Write JSON results to PATH (rank 0 only)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    pynccl_comm = None
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        gloo_group = dist.new_group(backend="gloo")
        comm = PyNcclCommunicator(group=gloo_group, device=device)
        if comm.available and not comm.disabled:
            pynccl_comm = comm
    except (ImportError, Exception):
        pass

    group = dist.group.WORLD
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"All-reduce dispatch benchmark")
        print(f"  World size: {world_size}")
        print(f"  Device: {torch.cuda.get_device_name(device)}")
        print(f"  pynccl available: {pynccl_comm is not None}")
        print(f"{'='*80}\n")

        header = f"{'nelems':>10} {'nbytes':>10}"
        if pynccl_comm is not None:
            header += f" | {'pynccl_wall':>12} {'pynccl_gr':>12}"
        header += f" | {'dist_event':>12} {'dist_graph':>12}"
        if pynccl_comm is not None:
            header += f" | {'wall_ratio':>12}"
        print(header)
        print("-" * len(header))

    for nelems in SIZES:
        tensor = torch.randn(nelems, dtype=torch.bfloat16, device=device)

        pynccl_result = None
        pynccl_graph_result = None
        if pynccl_comm is not None:
            pynccl_result = bench_pynccl(pynccl_comm, tensor.clone(),
                                         warmup=args.warmup, iters=args.iters)
            try:
                pynccl_graph_result = bench_cuda_graph(
                    lambda: setup_pynccl_graph(pynccl_comm, tensor.clone()))
            except Exception:
                pynccl_graph_result = {
                    "graph_throughput_us": float("nan"),
                    "graph_latency": {"p50_us": float("nan"), "p95_us": float("nan"),
                                      "min_us": float("nan"), "max_us": float("nan"), "iters": 0},
                }

        dist_result = bench_dist(group, tensor.clone(),
                                 warmup=args.warmup, iters=args.iters)
        try:
            dist_graph_result = bench_cuda_graph(
                lambda: setup_dist_graph(group, tensor.clone()))
        except Exception as e:
            if rank == 0:
                print(f"  CUDA graph capture FAILED for {nelems} elems: {e}")
            dist_graph_result = {
                "graph_throughput_us": float("nan"),
                "graph_latency": {"p50_us": float("nan"), "p95_us": float("nan"),
                                  "min_us": float("nan"), "max_us": float("nan"), "iters": 0},
            }

        if rank == 0:
            nbytes = nelems * 2
            line = f"{nelems:>10} {nbytes:>10}"

            if pynccl_result is not None:
                line += (
                    f" | {pynccl_result['gpu_wall']['p50_us']:>10.1f}us"
                    f" {pynccl_graph_result['graph_latency']['p50_us']:>10.1f}us"
                )

            line += (
                f" | {dist_result['gpu_event']['p50_us']:>10.1f}us"
                f" {dist_graph_result['graph_latency']['p50_us']:>10.1f}us"
            )

            if pynccl_result is not None:
                wall_ratio = (dist_result["gpu_wall"]["p50_us"]
                              / max(pynccl_result["gpu_wall"]["p50_us"], 0.1))
                line += f" | {wall_ratio:>11.2f}x"

            print(line)

            result = {
                "nelems": nelems,
                "nbytes": nbytes,
                "dist": {**dist_result, **dist_graph_result},
                "pynccl": None,
            }
            if pynccl_result is not None:
                result["pynccl"] = {**pynccl_result, **pynccl_graph_result}
            json_results.append(result)

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")

        if args.json:
            output = collect_metadata("allreduce_dispatch", tp=world_size, dtype="bf16")
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["pynccl_available"] = pynccl_comm is not None
            output["results"] = json_results
            write_json(args.json, output)

    if pynccl_comm is not None:
        pynccl_comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
