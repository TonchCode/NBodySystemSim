"""Summarise Nsight Systems CUDA traces (--trace=cuda --cuda-memory-usage=true).

Usage: python tools/memprofile_analysis.py <dir with *.sqlite>
For each trace: every device/pinned allocation (size, kind), the peak of live
device bytes over time, H2D/D2H/memset totals, and kernel time per kernel name.
"""
import sqlite3, sys, glob, os, re

def short(name):
    m = re.search(r"(\w+Kernel|DeviceRadixSort\w+Kernel)", name or "")
    return m.group(1) if m else (name or "?")[:40]

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.sqlite"))):
    c = sqlite3.connect(path)
    kinds = dict(c.execute("select id, label from ENUM_CUDA_MEM_KIND"))
    ops = dict(c.execute("select id, label from ENUM_CUDA_DEV_MEM_EVENT_OPER"))
    ev = c.execute("select start, bytes, memKind, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS order by start").fetchall()
    live = {}; peak = {}; allocs = {}
    for _, b, k, op in ev:
        kl, ol = kinds.get(k, k), ops.get(op, op)
        sign = 1 if "Alloc" in str(ol) else -1
        live[kl] = live.get(kl, 0) + sign * b
        peak[kl] = max(peak.get(kl, 0), live[kl])
        if sign > 0:
            allocs.setdefault(kl, []).append(b)
    print(f"\n=== {os.path.basename(path)[:-7]} ===")
    for kl in allocs:
        sz = sorted(allocs[kl], reverse=True)
        print(f"  {kl}: {len(sz)} allocations, peak live {peak[kl]/2**20:.1f} MiB; largest (MiB): "
              + ", ".join(f"{s/2**20:.2f}" for s in sz[:12]))
    mo = dict(c.execute("select id, label from ENUM_CUDA_MEMCPY_OPER"))
    for k, n, b in c.execute("select copyKind, count(*), sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind"):
        print(f"  memcpy {mo.get(k, k)}: {n} x, {b/2**20:.1f} MiB total")
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for n, b in c.execute("select count(*), sum(bytes) from CUPTI_ACTIVITY_KIND_MEMSET"):
            print(f"  memset: {n} x, {(b or 0)/2**20:.1f} MiB total")
    rows = c.execute("select s.value, count(*), sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value").fetchall()
    agg = {}
    for name, n, t in rows:
        a = agg.setdefault(short(name), [0, 0]); a[0] += n; a[1] += t
    tot = sum(v[1] for v in agg.values())
    for name, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  kernel {name:34s} {n:4d} x  avg {t/n/1e3:9.1f} us  {100*t/tot:5.1f} %")
