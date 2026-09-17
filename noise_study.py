#!/usr/bin/env python3
"""
Run-to-run noise of a benchmark on one build; thresholds and run counts
for regression gates follow from it.

  python noise_study.py run --nproc 8 --runs 10   # results under noise/
  python noise_study.py analyze noise/

`run` records a build fingerprint (torch commit + libtorch_cuda.so mtime)
and GPU clocks around each run; `analyze` discards runs whose build
changed. Per metric it reports run-to-run sigma, range, drift (second half
of runs vs first), and for a target change delta the runs per side an A/B
needs: (4 sigma / delta)^2 at 80% power, 5% significance. "3sig?" marks
delta >= 3 sigma, a rule of thumb for single-run gating. Sigma from a few
runs is rough (~40% at 4 runs); treat N as guidance.

bench_collectives: alpha (p50 at 1 KB), mid (p50 nearest 16 MB), beta
(algo GB/s at 1 GB), p95/p50 at both ends. Other benchmarks: every p50_us.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from compare_results import entry_key, extract_label, find_p50_metrics

MID_TARGET_BYTES = 16 << 20

# Smallest change worth flagging, % per metric.
DEFAULT_DELTA_PCT = {"alpha": 10.0, "mid": 5.0, "beta": 2.0,
                     "tail_small": 25.0, "tail_large": 25.0}


def fingerprint():
    # torch._C's directory holds the .so files for wheel and editable
    # installs alike; the mtime catches a same-commit rebuild.
    code = ("import os, torch, torch._C as C; "
            "lib = os.path.join(os.path.dirname(C.__file__), 'lib', 'libtorch_cuda.so'); "
            "print(torch.version.git_version, "
            "int(os.path.getmtime(lib)) if os.path.exists(lib) else 0)")
    try:
        return subprocess.check_output([sys.executable, "-c", code], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        sys.exit("cannot import torch to fingerprint the build; is it being rebuilt?")


def gpu_clocks():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True)
        return [line.strip() for line in out.strip().splitlines()]
    except Exception:
        return []


def cmd_run(args):
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = {"benchmark": args.bench, "nproc": args.nproc, "iters": args.iters,
                "dtype": args.dtype, "runs": []}
    for i in range(1, args.runs + 1):
        json_name = f"run_{i:02d}.json"
        json_path = os.path.join(args.out, json_name)
        rec = {"index": i, "json": json_name, "fingerprint": fingerprint(),
               "clocks_before": gpu_clocks(),
               "started": datetime.now(timezone.utc).isoformat()}
        # Only when given: some benchmarks take neither --iters nor --dtype.
        cmd = ["torchrun", f"--nproc_per_node={args.nproc}", f"{args.bench}.py",
               "--json", json_path]
        if args.iters is not None and "--iters" not in args.extra:
            cmd += ["--iters", str(args.iters)]
        if args.dtype and "--dtype" not in args.extra:
            cmd += ["--dtype", args.dtype]
        cmd += args.extra
        print(f"--- run {i}/{args.runs}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        rec.update(rc=rc, clocks_after=gpu_clocks(),
                   fingerprint_after=fingerprint())
        manifest["runs"].append(rec)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        if rc != 0:
            print(f"run {i} failed (rc={rc}); continuing", file=sys.stderr)
    print(f"manifest: {manifest_path}")


# ---- analysis ----

def collectives_metrics(data):
    """Tracked metrics for a bench_collectives JSON: {(collective, name): value}."""
    by_coll = {}
    for e in data.get("results", []):
        by_coll.setdefault(e["collective"], []).append(e)
    out = {}
    for coll, entries in by_coll.items():
        entries.sort(key=lambda e: e["nbytes"])
        small, large = entries[0], entries[-1]
        mid = min(entries, key=lambda e: abs(math.log(e["nbytes"] / MID_TARGET_BYTES)))
        out[(coll, "alpha")] = small["stats"]["p50_us"]
        out[(coll, "mid")] = mid["stats"]["p50_us"]
        out[(coll, "beta")] = large["algo_bw_gbps"]
        out[(coll, "tail_small")] = small["stats"]["p95_us"] / small["stats"]["p50_us"]
        out[(coll, "tail_large")] = large["stats"]["p95_us"] / large["stats"]["p50_us"]
    return out


def generic_metrics(data):
    out = {}
    for e in data.get("results", []):
        label = extract_label(e)
        for name, val in find_p50_metrics(e):
            out[(label, name)] = val
    return out


def load_runs(directory):
    manifest_path = os.path.join(directory, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else None
    runs = []
    if manifest:
        # Reference: the build after run 1, so a rebuild mid-run-1 discards
        # run 1, not every later run.
        ref = manifest["runs"][0].get("fingerprint_after", manifest["runs"][0]["fingerprint"])
        for rec in manifest["runs"]:
            path = os.path.join(directory, os.path.basename(rec["json"]))
            fps = {rec["fingerprint"], rec.get("fingerprint_after", rec["fingerprint"])}
            if rec.get("rc", 0) != 0:
                reason = f"failed (rc={rec['rc']})"
            elif fps != {ref}:
                reason = "build changed"
            elif not os.path.exists(path):
                reason = f"{path} missing"
            else:
                runs.append(json.load(open(path)))
                continue
            print(f"discarding run {rec['index']}: {reason}")
    else:
        for name in sorted(os.listdir(directory)):
            if name.startswith("run_") and name.endswith(".json"):
                runs.append(json.load(open(os.path.join(directory, name))))
        commits = {r.get("pytorch_commit") for r in runs}
        if len(commits) > 1:
            sys.exit(f"runs span several builds: {commits}")
    return runs


def cmd_analyze(args):
    runs = load_runs(args.dir)
    if len(runs) < 3:
        sys.exit(f"need at least 3 usable runs, have {len(runs)}")
    bench = runs[0].get("benchmark")
    extract = collectives_metrics if bench == "collectives" else generic_metrics
    series = {}
    for r in runs:
        for key, val in extract(r).items():
            series.setdefault(key, []).append(val)

    n = len(runs)
    half = n // 2
    incomplete = [key for key, vals in series.items() if len(vals) != n]
    if not series:
        sys.exit(f"{bench}: no p50_us metrics in these results (bench_verify "
                 f"reports pass/fail, not timings)")
    if incomplete:
        print(f"note: {len(incomplete)} metric(s) missing from some runs, not "
              f"tabulated: {', '.join(' '.join(k) for k in incomplete[:5])}"
              f"{' ...' if len(incomplete) > 5 else ''}")
    print(f"\n{bench}: {n} runs, build {runs[0].get('pytorch_commit', '?')[:12]}, "
          f"world_size {runs[0].get('world_size')}, dtype {runs[0].get('dtype')}")
    hdr = (f"{'metric':<40} {'median':>10} {'sigma%':>7} {'range%':>7} "
           f"{'drift%':>7} | {'delta%':>6} {'3sig?':>5} {'N':>3}  runs (sorted)")
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for key, vals in series.items():
        if len(vals) != n:
            continue
        med = statistics.median(vals)
        if med == 0:
            continue
        sigma = statistics.stdev(vals) / med * 100
        rng = (max(vals) - min(vals)) / med * 100
        drift = ((statistics.median(vals[half:]) - statistics.median(vals[:half]))
                 / med * 100)
        name = key[1]
        delta = args.delta.get(name, args.default_delta)
        ok = delta >= 3 * sigma
        runs_needed = max(1, math.ceil((4 * sigma / delta) ** 2)) if sigma > 0 else 1
        label = f"{key[0]} {name}"
        sorted_vals = " ".join(f"{v:.4g}" for v in sorted(vals))
        print(f"{label:<40} {med:>10.3f} {sigma:>7.2f} {rng:>7.2f} {drift:>+7.2f} "
              f"| {delta:>6.1f} {'yes' if ok else 'NO':>5} {runs_needed:>3}  {sorted_vals}")
        summary.append({"metric": label, "median": med, "sigma_pct": sigma,
                        "range_pct": rng, "drift_pct": drift, "delta_pct": delta,
                        "delta_ge_3sigma": ok, "runs_per_side": runs_needed,
                        "values": vals})
    print("\n  sigma%  = run-to-run stdev of the per-run value, % of median")
    print("  range%  = (max - min) / median")
    print("  drift%  = median of the last half of runs vs the first half; large "
          "values mean clocks/thermal drift, not noise")
    print("  delta%  = smallest change worth flagging (--delta name=pct to change)")
    print("  N       = runs per side for 80% power at 5% significance, (4 sigma/delta)^2")
    print("  runs    = the per-run values; two clusters mean a per-process state "
          "(e.g. an NCCL protocol choice), where sigma does not apply")
    out = os.path.join(args.dir, "noise_summary.json")
    with open(out, "w") as f:
        json.dump({"benchmark": bench, "runs": n, "metrics": summary}, f, indent=2)
    print(f"\nwritten {out}")


def parse_delta(items):
    d = dict(DEFAULT_DELTA_PCT)
    for item in items or []:
        name, sep, pct = item.partition("=")
        try:
            value = float(pct)
        except ValueError:
            value = -1.0
        if not sep or value <= 0:
            sys.exit(f"--delta expects NAME=PCT with PCT > 0, got {item!r}")
        d[name] = value
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark N times on the current build")
    r.add_argument("--bench", default="bench_collectives")
    r.add_argument("--nproc", type=int, required=True)
    r.add_argument("--runs", type=int, default=10)
    r.add_argument("--iters", type=int, default=None,
                   help="benchmark --iters; omit to use the benchmark's default")
    r.add_argument("--dtype", default=None,
                   help="benchmark --dtype; omit for the benchmark's default or for "
                        "benchmarks without one")
    r.add_argument("--out", default="noise")
    r.add_argument("extra", nargs="*", help="extra benchmark args after --")

    a = sub.add_parser("analyze", help="noise table from a run directory")
    a.add_argument("dir")
    a.add_argument("--delta", action="append", metavar="NAME=PCT",
                   help=f"override target delta; defaults {DEFAULT_DELTA_PCT}")
    a.add_argument("--default-delta", type=float, default=5.0,
                   help="delta for metrics not named in --delta (generic benchmarks)")

    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        args.delta = parse_delta(args.delta)
        cmd_analyze(args)


if __name__ == "__main__":
    main()
