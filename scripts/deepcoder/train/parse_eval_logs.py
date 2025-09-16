#!/usr/bin/env python3
import re, sys, csv, os, glob
from collections import OrderedDict

if len(sys.argv) != 2:
    print("Usage: parse_eval_logs.py <LOG_ROOT>", file=sys.stderr)
    sys.exit(2)

LOG_ROOT = sys.argv[1]
if not os.path.isdir(LOG_ROOT):
    print(f"Not a directory: {LOG_ROOT}", file=sys.stderr)
    sys.exit(2)

# Allow '@' in keys and capture simple "k: v" / "k=v"
PAIR = re.compile(r'([A-Za-z0-9_./:@-]+)\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

# Which exact keys to pull per mode → column aliases
TARGET_KEYS_BY_MODE = {
    'deterministic': {
        'val-core/codeforces/reward/mean@1':    'val_cf_reward@1',
        'val-core/livecodebench/reward/mean@1': 'val_lcb_reward@1',
        'val-core/areal/reward/mean@1':         'val_areal_reward@1',
    },
    'sample5': {
        'val-core/codeforces/reward/mean@5':    'val_cf_reward@5',
        'val-core/livecodebench/reward/mean@5': 'val_lcb_reward@5',
        'val-core/areal/reward/mean@5':         'val_areal_reward@5',
    },
}

rows = []
all_cols = set()

# Expect: LOG_ROOT/L{len}/{mode}/gs_{step}.log
for L_dir in sorted(glob.glob(os.path.join(LOG_ROOT, "L*"))):
    try:
        L = int(os.path.basename(L_dir)[1:])
    except Exception:
        continue

    for mode_dir in sorted(glob.glob(os.path.join(L_dir, "*"))):
        mode = os.path.basename(mode_dir)
        if not os.path.isdir(mode_dir):
            continue

        for log_file in sorted(glob.glob(os.path.join(mode_dir, "gs_*.log"))):
            base = os.path.basename(log_file)
            m = re.match(r'gs_(\d+)\.log', base)
            if not m:
                continue
            step = int(m.group(1))

            # Collect last occurrence of each key in the log
            last_vals = OrderedDict()
            try:
                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        for k, v in PAIR.findall(line):
                            try:
                                last_vals[k] = float(v)
                            except Exception:
                                continue
            except Exception as e:
                print(f"[WARN] Cannot read {log_file}: {e}", file=sys.stderr)
                continue

            row = OrderedDict()
            row['step'] = step
            row['max_prompt_len'] = L
            row['val_mode'] = mode

            # Pull only the metrics that correspond to this mode
            targets = TARGET_KEYS_BY_MODE.get(mode, {})
            for raw_key, alias in targets.items():
                if raw_key in last_vals:
                    row[alias] = last_vals[raw_key]
                    all_cols.add(alias)

            rows.append(row)

# Sort rows: prompt len → mode (deterministic first) → step
mode_order = {'deterministic': 0, 'sample5': 1}
rows.sort(key=lambda r: (r['max_prompt_len'], mode_order.get(r.get('val_mode',''), 9), r['step']))

# Header: fixed + any/all metric columns we observed
fixed = ['step', 'max_prompt_len', 'val_mode']
metric_cols = sorted(all_cols)  # will include @1 and/or @5 depending on logs present
header = fixed + metric_cols

def fmtf(x):
    if x is None:
        return ''
    if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.3e}"
    s = f"{x:.6f}".rstrip('0').rstrip('.')
    return s if s else "0"

# Column widths
col_w = {h: max(len(h), 8) for h in header}
for r in rows:
    for h in header:
        v = r.get(h, None)
        s = str(v) if h in ('step','max_prompt_len','val_mode') else (fmtf(v) if v is not None else '')
        col_w[h] = max(col_w[h], len(s))

# Print table
rule = sum(col_w.values()) + 2*(len(header)-1)
print()
print("RESULTS SUMMARY (reward)")
print("-" * rule)
print("  ".join(h.ljust(col_w[h]) for h in header))
print("-" * rule)
for r in rows:
    parts = []
    for h in header:
        if h in ('step','max_prompt_len','val_mode'):
            s = str(r.get(h, ''))
        else:
            v = r.get(h, None)
            s = fmtf(v) if v is not None else ''
        parts.append(s.ljust(col_w[h]))
    print("  ".join(parts))
print("-" * rule)
print(f"\nParsed {len(rows)} runs from: {LOG_ROOT}")

# CSV
csv_path = os.path.join(LOG_ROOT, "summary.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in rows:
        w.writerow([r.get(h, '') for h in header])

print(f"CSV saved: {csv_path}")
