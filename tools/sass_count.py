#!/usr/bin/env python3
"""Opcode histogram per kernel from a cuobjdump -sass dump (stdlib only).

    python tools/sass_count.py file.sass [--per 128]

--per K divides counts by K (e.g. the number of interactions in one fully unrolled tile)
to give instructions per interaction. Counts are STATIC (whole kernel, including the
prologue/epilogue and tile load), so with a fully unrolled 128-wide tile they overstate
the per-interaction loop cost by (prologue + tile-load instructions) / 128, typically < 0.3.
Opcode = the mnemonic without modifiers (FMUL.FTZ -> FMUL), predicates stripped.
"""
import argparse
import re
import sys
from collections import Counter


def parse(path):
    kernels, name, cnt = {}, None, None
    for line in open(path, errors="replace"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            name, cnt = m.group(1), Counter()
            kernels[name] = cnt
            continue
        m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?);", line)
        if m and cnt is not None:
            ins = re.sub(r"^@!?U?P[T0-9]\s+", "", m.group(1).strip())
            cnt[ins.split()[0].split(".")[0]] += 1
    return kernels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sass")
    ap.add_argument("--per", type=float, default=1.0)
    args = ap.parse_args()
    ks = parse(args.sass)
    ops = sorted({o for c in ks.values() for o in c}, key=lambda o: -sum(c[o] for c in ks.values()))
    names = list(ks)
    short = [re.sub(r".*?(k|Kernel)I(L[ib]\d+E)+E.*", lambda m: m.group(0), n) for n in names]
    print(f"# per = {args.per:g}  (counts / per; static whole-kernel counts)")
    for n in names:
        print(f"# kernel: {n}")
    hdr = f"{'opcode':<10}" + "".join(f"{'k' + str(i):>10}" for i in range(len(names)))
    print(hdr)
    for o in ops:
        print(f"{o:<10}" + "".join(f"{ks[n][o] / args.per:10.2f}" for n in names))
    print(f"{'TOTAL':<10}" + "".join(f"{sum(ks[n].values()) / args.per:10.2f}" for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
