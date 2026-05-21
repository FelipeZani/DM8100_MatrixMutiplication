#!/usr/bin/env python3
"""HPC Matrix Benchmarker: Evaluates strong and weak scaling for Serial, OpenMP, MPI, and CUDA.

Usage examples:
  python analysis/hpc_matrix_benchmarker.py --bins serialMux openMPMux mpiMux --mode strong \
      --sizes 256 512 1024 2056 4096 8096 --procs 1 2 4 8 --runs 3 --output-dir analysis/results

      
The script expects the compiled binaries to be in the repository root.
For OpenMP tests i need to set `OMP_NUM_THREADS`.
For MPI tests it uses `mpirun -n`.
"""
import argparse
import csv
import datetime
import json
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


class ExperimentRecorder:
    def __init__(self, out_dir, timestamp):
        self.out_dir = out_dir
        self.timestamp = timestamp
        self.log_path = os.path.join(out_dir, f"experiment_{timestamp}.log")
        self.csv_path = os.path.join(out_dir, f"raw_results_{timestamp}.csv")
        self.json_path = os.path.join(out_dir, f"results_{timestamp}.json")
        self.strong_summary_path = os.path.join(out_dir, f"strong_summary_{timestamp}.csv")
        self.weak_summary_path = os.path.join(out_dir, f"weak_summary_{timestamp}.csv")
        self.fields = [
            'binary', 'kind', 'size', 'procs', 'run', 'command',
            'parsed_time', 'raw_success', 'stdout'
        ]
        self._csv_file = None
        self._csv_writer = None
        self.records = []
        self._init_files()

    def _init_files(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fields)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def log(self, message, newline=True):
        text = f"{message}{'\n' if newline else ''}"
        sys.stdout.write(text)
        sys.stdout.flush()
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def record_run(self, record):
        record_copy = {k: record.get(k, '') for k in self.fields}
        self.records.append(record_copy)
        self._csv_writer.writerow(record_copy)
        self._csv_file.flush()
        try:
            os.fsync(self._csv_file.fileno())
        except OSError:
            pass

    def write_summary(self, rows, fieldnames, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def write_json(self, payload):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def close(self):
        if self._csv_file:
            self._csv_file.close()

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

def run_cmd(cmd, env=None, timeout=1200):
    try:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        return proc.stdout.decode(errors='ignore')
    except subprocess.TimeoutExpired:
        print(f"\n[ERROR] Command timed out after {timeout}s: {cmd}")
        return ''

def measure_binary(binname, size, procs=1, kind='serial', run_index=None, recorder=None):
    executable = binname if os.name == 'nt' or binname.startswith('./') or binname.startswith('.\\') else f"./{binname}"
    env = os.environ.copy()
    if kind == 'openmp':
        env['OMP_NUM_THREADS'] = str(procs)
        cmd = f"{executable} {size}"
    elif kind == 'mpi':
        cmd = f"mpirun -n {procs} {executable} {size}"
    else:
        cmd = f"{executable} {size}"

    out = run_cmd(cmd, env=env)
    t = parse_time(out)
    record = {
        'binary': binname,
        'kind': kind,
        'size': size,
        'procs': procs,
        'run': run_index,
        'command': cmd,
        'parsed_time': t,
        'raw_success': bool(out),
        'stdout': out.strip(),
    }
    if recorder is not None:
        recorder.record_run(record)
        recorder.log(f"Ran: {cmd}\nParsed time: {t}")
    else:
        print(f"Ran: {cmd}\nOutput:\n{out}\nParsed time: {t}")
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
    parser.add_argument('--mode', choices=['all', 'strong', 'weak'], default='all', help='Scaling mode to run (default: all)')
    parser.add_argument('--output-dir', default=os.path.join('analysis', 'results'), help='Directory to save logs, CSVs, and JSON results')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.abspath(args.output_dir)
    recorder = ExperimentRecorder(out_dir, timestamp)
    recorder.log(f"Experiment started at {datetime.datetime.now().isoformat()}")
    recorder.log(f"Output directory: {out_dir}")
    recorder.log(f"Arguments: {sys.argv[1:]}")

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
    if args.mode in ['all', 'strong']:
        recorder.log("Running strong-scaling experiments...")
        for binname in args.bins:
            kind = kind_map[binname]
            for size in args.sizes:
                # baseline (p=1) time
                base_t = None
                if 1 in args.procs:
                    tvals = []
                    for r in range(args.runs):
                        t, out = measure_binary(binname, size, procs=1, kind=kind, run_index=r, recorder=recorder)
                        if t is None:
                            recorder.log(f"Warning: couldn't parse time for {binname} size={size} run={r}\nOutput:\n{out}")
                        else:
                            tvals.append(t)
                    if tvals:
                        avg = sum(tvals)/len(tvals)
                        data[binname][(size,1)] = avg
                        base_t = avg
                    else:
                        data[binname][(size,1)] = None

                for p in args.procs:
                    if p == 1 and base_t is None:
                        continue
                    if p == 1:
                        continue
                    tvals = []
                    for r in range(args.runs):
                        t, out = measure_binary(binname, size, procs=p, kind=kind, run_index=r, recorder=recorder)
                        if t is None:
                            recorder.log(f"Warning: couldn't parse time for {binname} size={size} procs={p} run={r}\nOutput:\n{out}")
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

    strong_summary = []
    weak_summary = []

    # Print summarized strong-scaling table
    if args.mode in ['all', 'strong']:
        recorder.log('\nStrong-scaling summary (Speedup, Efficiency)')
        for binname in args.bins:
            recorder.log(f"\nBinary: {binname}")
            for size in args.sizes:
                base = data[binname].get((size,1))
                if isinstance(base, list):
                    if base:
                        base = sum(base)/len(base)
                    else:
                        base = None
                if base is None:
                    recorder.log(f" Size {size}: baseline (p=1) missing, skipping")
                    continue
                recorder.log(f" Size {size}: baseline T1 = {base:.6f} s")
                for p in sorted([x for x in args.procs if x>1]):
                    tp = data[binname].get((size,p))
                    if tp is None:
                        recorder.log(f"  p={p}: missing")
                        continue
                    speedup, eff = strong_scaling(tp, base, p)
                    if speedup is None:
                        recorder.log(f"  p={p}: no data")
                    else:
                        recorder.log(f"  p={p}: T={tp:.6f} s, speedup={speedup:.3f}, eff={eff*100:.1f}%")
                        strong_summary.append({
                            'binary': binname,
                            'size': size,
                            'procs': p,
                            'T': tp,
                            'speedup': speedup,
                            'efficiency': eff,
                        })

    # Weak scaling: use provided sizes mapped to processor counts
    if args.mode in ['all', 'weak']:
        recorder.log('\nWeak-scaling summary')
        for binname in args.bins:
            recorder.log(f"\nBinary: {binname}")
            sorted_procs = sorted(args.procs)
            # For single-GPU CUDA, weak scaling with 'procs' > 1 is not meaningful.
            # The binary itself doesn't use the 'procs' argument.
            if kind_map[binname] == 'cuda' and any(p > 1 for p in sorted_procs):
                recorder.log(f"  Skipping weak-scaling for CUDA binary {binname} as it's a single-GPU implementation.")
            for i, p in enumerate(sorted_procs):
                if i < len(args.sizes):
                    target_size = args.sizes[i]
                else:
                    target_size = weak_size_for(args.sizes[0], 1, p)
                t = data[binname].get((target_size,p))
                # If we retrieved a list (legacy check) or None, try to run it
                if kind_map[binname] == 'cuda' and p > 1:
                    # If we skipped above, ensure this is also skipped for data collection
                    recorder.log(f"  Skipping data collection for {binname} (CUDA) with p={p} (single-GPU).")
                    continue
                if isinstance(t, list):
                    t = (sum(t)/len(t)) if t else None
                
                if t is None:
                    tvals = []
                    for r in range(args.runs):
                        tt, out = measure_binary(binname, target_size, procs=p, kind=kind_map[binname], run_index=r, recorder=recorder)
                        if tt is None:
                            recorder.log(f"Warning: couldn't parse time for weak-scaling {binname} size={target_size} procs={p} run={r}\nOutput:\n{out}")
                        else:
                            tvals.append(tt)
                    t = (sum(tvals)/len(tvals)) if tvals else None
                if t is None:
                    recorder.log(f" p={p}: size={target_size}: missing")
                    continue
                recorder.log(f" p={p}: size={target_size}: T={t:.6f} s")
                weak_summary.append({
                    'binary': binname,
                    'procs': p,
                    'size': target_size,
                    'T': t,
                })

    if strong_summary:
        recorder.write_summary(strong_summary, ['binary', 'size', 'procs', 'T', 'speedup', 'efficiency'], recorder.strong_summary_path)
    if weak_summary:
        recorder.write_summary(weak_summary, ['binary', 'procs', 'size', 'T'], recorder.weak_summary_path)

    recorder.write_json({
        'args': vars(args),
        'records': recorder.records,
        'strong_summary': strong_summary,
        'weak_summary': weak_summary,
    })
    recorder.log('\nFinished. Saved results to:')
    recorder.log(f"  log: {recorder.log_path}")
    recorder.log(f"  raw CSV: {recorder.csv_path}")
    if strong_summary:
        recorder.log(f"  strong summary CSV: {recorder.strong_summary_path}")
    if weak_summary:
        recorder.log(f"  weak summary CSV: {recorder.weak_summary_path}")
    recorder.log(f"  JSON: {recorder.json_path}")
    recorder.close()
    

if __name__ == '__main__':
    main()
