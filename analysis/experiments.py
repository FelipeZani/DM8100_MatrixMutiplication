#!/usr/bin/env python3
"""Run scaling experiments for the matrix-multiplication binaries and summarize strong/weak scaling.

Usage examples:
  python analysis/experiments.py --bins serialMux openMPMux mpiMux \
      --sizes 256 512 1024 2056 4096 8096 --procs 1 2 4 8 --runs 3

The script expects the compiled binaries to be in the repository root.
For OpenMP tests i need to set `OMP_NUM_THREADS`.
For MPI tests it uses `mpirun -n`.
"""
import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict

TIME_RE = re.compile(r"Exec Time: *([0-9]+\.?[0-9eE+-]*)") 
TIME_RE_ALT = re.compile(r"Exec Time: *([0-9]+\.?[0-9eE+-]*) s") 
TIME_RE_ANY = re.compile(r"([0-9]+\.?[0-9eE+-]*)\s*s") 

def parse_time(output):
    # Try several common patterns printed by the programs
    for regex in (TIME_RE, TIME_RE_ALT, TIME_RE_ANY):
        m = regex.search(output)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    # Try to find 'Exec Time' followed by number in same line
    for line in output.splitlines():
        if 'Exec Time' in line or 'Exec Time:' in line or 'Exec Time' in line:
            nums = re.findall(r"[0-9]+\.?[0-9eE+-]*", line)
            if nums:
                return float(nums[-1])
    return None

def run_cmd(cmd, env=None, timeout=600):
    try:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        return proc.stdout.decode(errors='ignore')
    except subprocess.TimeoutExpired:
        return ''

def measure_binary(binname, size, procs=1, kind='serial'):
    cwd = os.getcwd()
    cmd = None
    env = os.environ.copy()
    if kind == 'openmp':
        env['OMP_NUM_THREADS'] = str(procs)
        cmd = f"./{binname} {size}"
    elif kind == 'mpi':
        cmd = f"mpirun -n {procs} ./{binname} {size}"
    else:
        cmd = f"./{binname} {size}"

    out = run_cmd(cmd, env=env)
    t = parse_time(out)
    return t, out

def strong_scaling(results, base_time, procs):
    # speedup = T1 / Tp ; efficiency = speedup / p
    speedup = base_time / results if results and base_time else None
    eff = (speedup / procs) if speedup is not None else None
    return speedup, eff

def weak_size_for(base_size, base_procs, procs):
    # For cubic work (N^3), keep N proportional to p^(1/3)
    ratio = (procs / base_procs) ** (1.0/3.0)
    return max(1, int(round(base_size * ratio)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', nargs='+', required=True, help='Binaries to test (serialMux/openMPMux/mpiMux/cudaMux)')
    parser.add_argument('--sizes', nargs='+', type=int, required=True, help='Matrix sizes N to test')
    parser.add_argument('--procs', nargs='+', type=int, default=[1,2,4,8], help='Processor/thread counts to test')
    parser.add_argument('--runs', type=int, default=3, help='Repetitions per configuration')
    parser.add_argument('--skip-missing', action='store_true', help='Skip missing binaries instead of failing')
    args = parser.parse_args()

    # classify binaries
    kind_map = {}
    for b in args.bins:
        if 'openmp' in b.lower() or 'omp' in b.lower():
            kind_map[b] = 'openmp'
        elif 'mpi' in b.lower():
            kind_map[b] = 'mpi'
        elif 'cuda' in b.lower():
            kind_map[b] = 'cuda'
        else:
            kind_map[b] = 'serial'

    # verify binaries exist
    missing = [b for b in args.bins if not os.path.isfile(os.path.join(os.getcwd(), b))]
    if missing:
        msg = f"Missing binaries: {missing}"
        if args.skip_missing:
            print(msg + ". They will be skipped.")
            args.bins = [b for b in args.bins if b not in missing]
        else:
            print(msg)
            sys.exit(1)

    data = defaultdict(lambda: defaultdict(list))

    # Strong scaling: for each fixed size, vary procs
    print("Running strong-scaling experiments...")
    for binname in args.bins:
        kind = kind_map[binname]
        for size in args.sizes:
            # baseline (p=1) time
            base_t = None
            if 1 in args.procs:
                tvals = []
                for r in range(args.runs):
                    t, out = measure_binary(binname, size, procs=1, kind=kind)
                    if t is None:
                        print(f"Warning: couldn't parse time for {binname} size={size} run={r}\nOutput:\n{out}")
                    else:
                        tvals.append(t)
                if tvals:
                    base_t = sum(tvals)/len(tvals)
                data[binname][(size,1)] = tvals

            for p in args.procs:
                if p == 1 and base_t is None:
                    continue
                if p == 1:
                    continue
                tvals = []
                for r in range(args.runs):
                    t, out = measure_binary(binname, size, procs=p, kind=kind)
                    if t is None:
                        print(f"Warning: couldn't parse time for {binname} size={size} procs={p} run={r}\nOutput:\n{out}")
                    else:
                        tvals.append(t)
                if tvals:
                    avg = sum(tvals)/len(tvals)
                    data[binname][(size,p)] = avg
                else:
                    data[binname][(size,p)] = None

    # Print summarized strong-scaling table
    print('\nStrong-scaling summary (Speedup, Efficiency)')
    for binname in args.bins:
        print(f"\nBinary: {binname}")
        for size in args.sizes:
            base = data[binname].get((size,1))
            if isinstance(base, list):
                if base:
                    base = sum(base)/len(base)
                else:
                    base = None
            if base is None:
                print(f" Size {size}: baseline (p=1) missing, skipping")
                continue
            print(f" Size {size}: baseline T1 = {base:.6f} s")
            for p in sorted([x for x in args.procs if x>1]):
                tp = data[binname].get((size,p))
                if tp is None:
                    print(f"  p={p}: missing")
                    continue
                speedup, eff = strong_scaling(tp, base, p)
                if speedup is None:
                    print(f"  p={p}: no data")
                else:
                    print(f"  p={p}: T={tp:.6f} s, speedup={speedup:.3f}, eff={eff*100:.1f}%")

    # Weak scaling: keep per-core work approx constant; scale N ~ p^(1/3)
    print('\nWeak-scaling summary (normalized to base size)')
    # choose base_size from first size in args.sizes
    base_size = args.sizes[0]
    base_procs = 1
    for binname in args.bins:
        print(f"\nBinary: {binname}")
        for p in sorted(args.procs):
            target_size = weak_size_for(base_size, base_procs, p)
            # measure or reuse if available
            t = data[binname].get((target_size,p))
            if t is None:
                # try to run
                tvals = []
                for r in range(args.runs):
                    tt, out = measure_binary(binname, target_size, procs=p, kind=kind_map[binname])
                    if tt is not None:
                        tvals.append(tt)
                t = (sum(tvals)/len(tvals)) if tvals else None
            if t is None:
                print(f" p={p}: size={target_size}: missing")
                continue
            print(f" p={p}: size={target_size}: T={t:.6f} s")

    print('\nFinished. Save or extend this script to produce plots or CSV output as needed.')

if __name__ == '__main__':
    main()
