---
name: impact-report
description: Generate impact reports attributing benchmark improvements to specific team contributions. Use when the user mentions impact, contribution, improvement, attribution, report, summary, team impact, show value, demonstrate, what improved, release notes, or needs to quantify the distributed team's measurable effect on product performance.
---

# Impact Report

Generate reports that connect pytorch-dist-bench data to team contributions. The audience is engineering leadership and partner teams who need measurable impact, not raw benchmark numbers.

## Report Structure

Every impact report has four sections:

### 1. Executive Summary

One paragraph: what changed, by how much, and what it means for the product.

Example: "Between RHAII 2.0 (PyTorch 2.6, NCCL 2.21) and RHAII 2.1 (PyTorch 2.8, NCCL 2.25), TP inference AllReduce latency improved 18% at decode-phase tensor sizes and FSDP2 training step time improved 12%. These improvements directly reduce vLLM token-generation latency and InstructLab fine-tuning wall time."

### 2. Attribution Table

Each improvement maps to a specific change with its category:

| Benchmark | Metric | Change | Cause | Category |
|---|---|---|---|---|
| `bench_inference_tp_vllm` | AllReduce p50 (decode, Llama-70B) | -18% | NCCL 2.21 → 2.25 tree algorithm for small messages | NCCL version |
| `bench_fsdp2_training` | Step time p50 (8 layers, batch=4) | -12% | PyTorch PR #187642: FSDP2 overlap scheduling | Upstream contribution |
| `bench_collectives` | AllReduce bus_bw (1GB) | +8% | NCCL NVLS algorithm enabled by default | NCCL version |

### 3. Benchmark-to-Product Mapping

Which benchmarks correspond to which product workloads:

| Benchmark | Product workload | What it measures |
|---|---|---|
| `bench_inference_tp_vllm` | vLLM serving (RHOAI) | TP AllReduce on the critical path for every decode token |
| `bench_collectives` | All distributed workloads | Raw NCCL performance underpinning everything |
| `bench_fsdp2_training` | InstructLab, fine-tuning | FSDP2 training step including overlap scheduling |
| `bench_pipeline_parallel` | Large model training (PP+DP) | P2P bandwidth and pipeline step efficiency |
| `bench_compile_distributed` | Any compiled workload | torch.compile interaction with distributed ops |
| `bench_multinode` | Multi-node clusters | IB/RoCE scale-out efficiency |
| `bench_moe_alltoall` | MoE model serving/training | Expert dispatch overhead |
| `bench_fp8_fused_ops` | FP8 inference/training | Quantized fused collective performance |
| `bench_symm_mem_fused_ops` | TP inference (fused path) | NVSwitch-accelerated fused GEMM+collective |
| `bench_training_fsdp_collectives` | FSDP2 training (raw ops) | AllGather/ReduceScatter at FSDP shard sizes |
| `bench_allreduce_dispatch` | All distributed workloads | Python/C++ dispatch overhead per collective call |
| `bench_migration_path` | Library migration decisions | pynccl → dist → fused ops progression |

### 4. Methodology and Caveats

Always include. This section protects the report's credibility.

## How to Generate

### Step 1: Produce comparison data

```bash
python compare_results.py results/version_a/ results/version_b/ --threshold 3
```

### Step 2: Extract improvements

Filter the output for `IMPROVED` lines (negative % change = faster). Record the benchmark, metric, baseline value, test value, and % change.

### Step 3: Identify causal changes

For each improvement, determine what changed between the two versions:

- **PyTorch version delta**: `git log --oneline v2.6.0..v2.8.0 -- torch/distributed/` in the PyTorch repo
- **NCCL version delta**: NCCL release notes between the two versions
- **Driver/firmware changes**: Compare `nvidia-smi` output from both runs (captured in JSON metadata)
- **RHAII-specific patches**: Internal patch list between releases

### Step 4: Classify each improvement

Categories:
- **Upstream PyTorch contribution** — a PR the team authored or reviewed that landed in PyTorch
- **NCCL version/tuning** — NCCL upgrade or env var tuning the team recommended
- **Driver/firmware** — GPU driver or NVSwitch firmware update
- **RHAII-specific patch** — patch carried in RHAII but not yet upstream
- **Infrastructure** — hardware, network, or cluster configuration change

### Step 5: Write the report

Use the structure above. Include the caveats below.

## Accuracy Requirements

### Benchmark improvement vs product improvement

A "20% AllReduce improvement" is not a "20% product improvement." AllReduce is one operation in a larger pipeline. The product impact depends on what fraction of end-to-end time the operation represents.

When you know the fraction: "AllReduce improved 20%, which represents ~15% of vLLM decode time, yielding an estimated ~3% end-to-end improvement."

When you don't: "AllReduce improved 20% in isolation. End-to-end impact depends on the communication/compute ratio of the specific workload, which this benchmark does not measure."

Never equate the benchmark improvement with the product improvement unless end-to-end data confirms it.

### Attribution requires isolation

- **`ab_test_pytorch_pr.sh` result** — one variable changed, everything else held constant. Direct attribution is valid: "PR #187642 improved FSDP2 step time by 12%."
- **Release-over-release comparison** — multiple variables changed. Attribution requires decomposition: test each change in isolation to identify which one drove the improvement.

Language:
- Single-variable A/B: "caused by", "resulted from", "due to"
- Multi-variable comparison: "correlated with", "coincided with", "associated with the upgrade from X to Y"

### Measurement confidence

For numbers in reports or presentations:
- Run each configuration at least 3 times independently
- Report the median across runs, not a single run
- Include the range: "improved 18% (range: 15-21% across 5 runs)"
- Lock GPU clocks for reproducibility

A single-run delta that falls within the IQR is indistinguishable from noise. Do not report it.

## Narrative Templates

### For engineering leadership

Focus on product impact and competitive position:

> "The distributed team's contributions to NCCL tuning and FSDP2 overlap scheduling delivered measurable improvements in RHAII 2.1: vLLM decode latency reduced by X% and training throughput increased by Y%. These improvements were validated on 8×H200 systems using pytorch-dist-bench, our distributed performance regression suite."

### For upstream community

Focus on the specific changes and their measured effect:

> "PR #187642 (FSDP2 AllGather/compute overlap rescheduling) reduced FSDP2 training step time by 12% on 8×H200 NVSwitch, measured by pytorch-dist-bench at Llama-70B dimensions (hidden=8192, intermediate=28672, 8 layers, batch=4). The improvement comes from better AllGather prefetching that hides more communication behind backward compute."

### For QA / release notes

Focus on what was tested, on what hardware, with what methodology:

> "Distributed performance validated on 8×H200 NVSwitch using pytorch-dist-bench (12 single-node benchmarks, 200 iterations each, GPU clocks locked at X MHz). Key metrics: AllReduce p50 [value]us (±[iqr]us), FSDP2 step p50 [value]us (±[iqr]us). No regressions detected above 5% threshold. Full JSON results archived at [path]."
