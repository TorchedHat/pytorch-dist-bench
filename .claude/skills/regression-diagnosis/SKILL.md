---
name: regression-diagnosis
description: Diagnose a reported product performance regression using pytorch-dist-bench benchmark evidence. Use when the user mentions regression, slowdown, slower, degraded, performance drop, diagnose, root cause, investigation, or when a product team reports that inference or training got slower on a new release.
---

# Regression Diagnosis

Use the pytorch-dist-bench suite to narrow a reported product regression to a root cause. The benchmarks decompose the distributed stack into isolated layers; work through them to identify which layer regressed.

## The Diagnostic Ladder

Work top-down. Each step either identifies the cause or narrows the search.

### Step 0: Get baseline and test results

Run the full suite on both versions and compare:

```bash
# On baseline version
./run_all.sh 8 --json-dir results/baseline/

# On test version
./run_all.sh 8 --json-dir results/test/

# Compare
python compare_results.py results/baseline/ results/test/ --threshold 5
```

Read the `compare_results.py` output. Regressions are flagged with `REGRESSION` (p50_us increased beyond threshold). Note which benchmarks regressed and which didn't — the pattern matters more than individual numbers.

### Step 1: Communication or compute?

Compare `bench_inference_tp_vllm` results:

- **TP layer regressed but raw AllReduce at same tensor sizes didn't** → compute regression (GEMM kernels, activation functions). Points at PyTorch/CUDA/cuBLAS version change.
- **Raw AllReduce regressed** → communication regression. Go to Step 2.
- **Both regressed proportionally** → system-level issue (GPU clocks, driver, power policy). Check `nvidia-smi -q -d CLOCK,POWER`.

For training regressions, compare `bench_fsdp2_training` step time against `bench_collectives` AllGather/ReduceScatter at matching tensor sizes.

### Step 2: Where in the communication stack?

- **`bench_collectives`** AllReduce/AllGather/ReduceScatter regressed across sizes → NCCL itself is slower. Check NCCL version change.
  - Only at small sizes (<256KB) → latency regression, likely algorithm/protocol selection. Check `NCCL_ALGO`, `NCCL_PROTO`.
  - Only at large sizes (>4MB) → bandwidth regression. Check NVLink/IB link health, `NCCL_MAX_NCHANNELS`.
  - Across all sizes → fundamental transport issue.
- **`bench_allreduce_dispatch`** ProcessGroup overhead increased → PyTorch `torch.distributed` layer regression above NCCL.
- **`bench_collectives` fine but `bench_fsdp2_training` regressed** → FSDP2 overlap scheduling broke (AllGather/ReduceScatter timing relative to compute changed). Points at `torch.distributed.fsdp` code change.

**Important**: Isolated collective benchmarks run one operation at a time. Real workloads overlap communication with compute, competing for GPU resources (SMs, memory bandwidth, L2 cache). If `bench_collectives` shows no regression but `bench_fsdp2_training` or `bench_compile_distributed` does, the regression is in how communication and compute interact under contention, not in raw NCCL performance. Check whether FSDP2 overlap scheduling or torch.compile's communication placement changed.

### Step 3: Which parallelism pattern?

- **`bench_pipeline_parallel`** P2P regressed but collectives didn't → P2P transport path is different from collective path in NCCL. Check `NCCL_P2P_LEVEL`.
- **`bench_compile_distributed`** compiled path regressed, eager fine → Inductor distributed codegen regression.
- **`bench_moe_alltoall`** regressed → All-to-All specifically affected. Different algorithm from AllReduce.

### Step 4: Single-node vs multi-node?

If you have multi-node results from `bench_multinode`:

- **`intra_node` fine, `inter_node` regressed** → IB/RoCE driver, firmware, or NCCL net plugin.
- **`inter_node` fine, `inter_agg` regressed** → contention handling changed (multiple DP streams competing for IB bandwidth).
- **`intra_node` regressed** → NVLink issue (driver, NVSwitch firmware).

### Step 5: Negative evidence

When benchmarks DON'T show a regression, that narrows the search. But state the boundary:

Negative evidence only covers tested operations and message sizes. The suite sweeps 11 sizes (512 to 536M elements) and tests AllReduce, AllGather, ReduceScatter, P2P Send/Recv, and All-to-All. If the product uses a collective pattern, message size, or process group topology not in the sweep, absence of regression here does not rule it out. When reporting negative evidence, always state: "No regression observed in [specific operations tested]. Operations not covered: [list what's missing]."

## Evidence Table

| Benchmark evidence | Root cause | Owner |
|---|---|---|
| `bench_collectives` AllReduce regressed at all sizes | NCCL regression | NCCL version pin or env tuning |
| `bench_collectives` regressed only at small sizes | NCCL latency (algorithm selection) | NCCL tuning (`NCCL_ALGO`, `NCCL_PROTO`) |
| `bench_collectives` fine, `bench_inference_tp_vllm` TP layer regressed | cuBLAS/GEMM kernel regression | PyTorch/CUDA version |
| `bench_allreduce_dispatch` ProcessGroup overhead increased | PyTorch distributed layer | PyTorch version pin |
| `bench_collectives` fine, `bench_fsdp2_training` regressed | FSDP2 overlap scheduling | PyTorch FSDP code |
| `bench_compile_distributed` compiled regressed, eager fine | Inductor codegen | PyTorch compiler team |
| `bench_multinode` inter_node regressed, intra_node fine | IB driver/firmware/NCCL net plugin | Infra/networking |
| `bench_pipeline_parallel` P2P regressed, collectives fine | NCCL P2P transport | NCCL P2P config |
| No benchmark regressed | Regression is outside distributed communication | Application/serving layer |

## Commands at Each Step

```bash
# Step 0: Full comparison
python compare_results.py results/baseline/ results/test/

# Step 1: Isolate TP inference communication vs compute
torchrun --nproc_per_node=8 bench_inference_tp_vllm.py --section allreduce --json results/tp_allreduce.json
torchrun --nproc_per_node=8 bench_inference_tp_vllm.py --section layer --json results/tp_layer.json

# Step 2: Raw collectives
torchrun --nproc_per_node=8 bench_collectives.py --json results/collectives.json

# Step 2: Dispatch overhead
torchrun --nproc_per_node=8 bench_allreduce_dispatch.py --json results/dispatch.json

# Step 3: Specific patterns
torchrun --nproc_per_node=8 bench_pipeline_parallel.py --section p2p --json results/p2p.json
torchrun --nproc_per_node=8 bench_compile_distributed.py --json results/compile.json

# Step 4: Multi-node decomposition
torchrun --nnodes=N --nproc_per_node=G --rdzv_backend=c10d --rdzv_endpoint=MASTER:29500 \
  bench_multinode.py --json results/multinode.json
```
