#!/usr/bin/env python3
"""Clean re-run analysis (docs/predictions/2026-10-18-clean-rerun.md). Stdlib only.

    python tools/clean_rerun_analysis.py results/2026-10-18_clean_rerun

Steps (the comparison rule of the pre-registration, §2):
 1. tree round 2: tools/tree_analysis.py --round2 logic run on pass 1, pass 2 and on the MERGED pass
    (every time = mean of the two pass medians; rates = harmonic mean, i.e. counted work / mean time;
    timeline metrics = mean of the two passes). Ranges are the committed ones (c8d9fcc), unchanged.
 2. IC round: tools/ic_analysis.py run on pass 1, pass 2 and on merged CSVs (same rule; be68419 ranges).
 3. scorecard.csv: item, original prediction range, old value, clean value, ratio clean/old, old verdict,
    clean verdict, pass-1 and pass-2 verdicts, category (primary / secondary / identity), stability.
 4. load_tests.txt: the seven falsification tests of §5 and the clean/old ratio tables of §3.
"""
import csv, glob, io, math, os, re, shutil, statistics, subprocess, sys
from contextlib import redirect_stdout

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import tree_analysis as ta  # noqa: E402
import timeline_analysis as tla  # noqa: E402

R = sys.argv[1]
OLD_T2 = 'results/2026-10-17_tree_round2'
OLD_IC = 'results/2026-10-17_ic_round'
OLD_T1 = 'results/2026-10-17_tree_round1'
TIME_KEYS = ['trav', 'total_ms', 'keys', 'sort', 'tree', 'order', 'build']
RATE_KEYS = ['rate', 'rate_real', 'rate_trav', 'rate_ap']


# ------------------------------------------------------------------ tree round 2, merged pass
def merged_parse(path):
    """parse_log2 on p1 and p2 copies of a log; times averaged, rates harmonic-averaged."""
    name = os.path.basename(path)
    p1 = ta._orig_parse_log2(os.path.join(R, 'tree2', 'p1', 'logs', name))
    p2p = os.path.join(R, 'tree2', 'p2', 'logs', name)
    if p1 is None or not os.path.exists(p2p):
        return p1
    p2 = ta._orig_parse_log2(p2p)
    if p2 is None:
        return p1
    m = dict(p1)
    for k in TIME_KEYS:
        if k in p1 and k in p2:
            m[k] = 0.5 * (p1[k] + p2[k])
    for k in RATE_KEYS:
        if k in p1 and k in p2 and p1[k] > 0 and p2[k] > 0:
            m[k] = 2.0 / (1.0 / p1[k] + 1.0 / p2[k])
    m['trav_p1'], m['trav_p2'] = p1.get('trav'), p2.get('trav')
    m['gates'] = 'FAIL' if 'FAIL' in (p1['gates'], p2['gates']) else p1['gates']
    return m


def merged_slot(path):
    name = os.path.basename(path)
    a, meta = ta._orig_slot(os.path.join(R, 'tree2', 'p1', 'timeline', name))
    p2 = os.path.join(R, 'tree2', 'p2', 'timeline', name)
    if not os.path.exists(p2):
        return a, meta
    b, _ = ta._orig_slot(p2)
    return {k: 0.5 * (a[k] + b[k]) for k in a}, meta


def tree_passes():
    out = {}
    for P in ['p1', 'p2']:
        d = os.path.join(R, 'tree2', P)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ta.round2(d)
        open(os.path.join(d, 'analysis_output.txt'), 'w').write(buf.getvalue())
        out[P] = d
    # merged: logs/ and timeline/ hold name placeholders (p1 copies); the parsers read p1 + p2
    md = os.path.join(R, 'tree2', 'merged')
    for sub in ['logs', 'timeline']:
        os.makedirs(os.path.join(md, sub), exist_ok=True)
        for f in glob.glob(os.path.join(R, 'tree2', 'p1', sub, '*')):
            if f.endswith('index.txt'):
                continue
            shutil.copy(f, os.path.join(md, sub, os.path.basename(f)))
    ta._orig_parse_log2, ta._orig_slot = ta.parse_log2, ta.slot_metrics
    ta.parse_log2, ta.slot_metrics = merged_parse, merged_slot
    buf = io.StringIO()
    with redirect_stdout(buf):
        ta.round2(md)
    ta.parse_log2, ta.slot_metrics = ta._orig_parse_log2, ta._orig_slot
    open(os.path.join(md, 'analysis_output.txt'), 'w').write(buf.getvalue())
    out['merged'] = md
    return out


# ------------------------------------------------------------------ IC round, merged pass
def ic_timelines(d):
    files = sorted(glob.glob(os.path.join(d, 'timeline', 'T_*.csv')))
    if files:
        r = subprocess.run([sys.executable, os.path.join(TOOLS, 'timeline_analysis.py')] + files +
                           ['--summary', os.path.join(d, 'timeline_metrics.csv')], capture_output=True, text=True)
        open(os.path.join(d, 'timeline_analysis_output.txt'), 'w').write(r.stdout + r.stderr)


def merge_csv(name, out_dir):
    p1 = os.path.join(R, 'ic', 'p1', name)
    p2 = os.path.join(R, 'ic', 'p2', name)
    if not os.path.exists(p1):
        return
    r1 = list(csv.DictReader(open(p1)))
    r2 = list(csv.DictReader(open(p2))) if os.path.exists(p2) else []
    key = lambda r: (r['kernel'], r['n'], r['ic'], r['leaf'], r['block'], r['mass_spectrum'], r['run'])
    k2 = {key(r): r for r in r2}
    out = []
    for r in r1:
        m = dict(r)
        s = k2.get(key(r))
        if s:
            for c in r:
                if c.startswith('ph_') or c == 'ms':
                    m[c] = repr(0.5 * (float(r[c]) + float(s[c])))
                elif c in ('trav_ginter_per_s', 'ginteractions_per_s') and float(r[c]) > 0 and float(s[c]) > 0:
                    m[c] = repr(2.0 / (1.0 / float(r[c]) + 1.0 / float(s[c])))
        out.append(m)
    with open(os.path.join(out_dir, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(r1[0].keys()))
        w.writeheader(); w.writerows(out)


def ic_passes(viewer_rate):
    for P in ['p1', 'p2']:
        ic_timelines(os.path.join(R, 'ic', P))
    md = os.path.join(R, 'ic', 'merged')
    os.makedirs(os.path.join(md, 'logs'), exist_ok=True)
    for f in glob.glob(os.path.join(R, 'ic', 'p1', 'logs', '*.txt')):
        shutil.copy(f, os.path.join(md, 'logs', os.path.basename(f)))
    for n in ['setA.csv', 'setB.csv', 'setK.csv']:
        merge_csv(n, md)
    t1, t2 = (os.path.join(R, 'ic', P, 'timeline_metrics.csv') for P in ['p1', 'p2'])
    if os.path.exists(t1) and os.path.exists(t2):
        a = list(csv.DictReader(open(t1))); b = {r['file']: r for r in csv.DictReader(open(t2))}
        for r in a:
            s = b.get(r['file'])
            if s:
                for c in ('sm_eff', 'tail_frac', 'max_over_mean', 'span_us', 'mean_busy_us'):
                    r[c] = repr(0.5 * (float(r[c]) + float(s[c])))
        with open(os.path.join(md, 'timeline_metrics.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(a[0].keys())); w.writeheader(); w.writerows(a)
    for d in [os.path.join(R, 'ic', 'p1'), os.path.join(R, 'ic', 'p2'), md]:
        args = [sys.executable, os.path.join(TOOLS, 'ic_analysis.py'), d]
        if viewer_rate is not None:
            args += ['--viewer-rate', str(viewer_rate)]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode:
            print('ic_analysis failed on', d, r.stderr[-2000:])
    return md


# ------------------------------------------------------------------ scorecard
def nums(s):
    return [float(x) for x in re.findall(r'(?<![A-Za-z_0-9])[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', re.sub(r'\w+=', ' ', s))]


def ratio_str(old, new):
    a, b = nums(old), nums(new)
    if not a or len(a) != len(b):
        return 'n/a'
    rr = [y / x if x else float('nan') for x, y in zip(a, b)]
    return '; '.join(f'{v:.4f}' for v in rr)


def read_sc(path, idk, verk):
    if not os.path.exists(path):
        return {}
    return {r[idk]: r for r in csv.DictReader(open(path))}


T2_PRIMARY = ['P2-P%d' % i for i in range(1, 12)] + ['P2-P13'] + ['P2-W%d' % i for i in range(8, 13)] + \
             ['P2-L5'] + ['P2-X%d' % i for i in range(1, 6)]
T2_SECONDARY = ['P2-C1', 'P2-C2', 'P2-C3', 'P2-C4', 'P2-C5', 'P2-W7', 'P2-W13', 'P2-W16', 'P2-W17', 'P2-W18', 'P2-D1',
                'P2-D2', 'P2-L1', 'P2-L2', 'P2-L3', 'P2-L4']


def category_t2(i):
    if i.startswith('P2-O'):
        return None
    if i in T2_PRIMARY:
        return 'primary'
    if i in T2_SECONDARY:
        return 'secondary'
    return 'identity'


def category_ic(i):
    if i.startswith('S') or i.startswith('Q') or i in ('K3', 'V2'):
        return 'primary'
    return 'identity'


def build_scorecard(td, icd):
    rows = []
    old = read_sc(os.path.join(OLD_T2, 'scorecard.csv'), 'id', 'verdict')
    sc = {P: read_sc(os.path.join(R, 'tree2', P, 'scorecard.csv'), 'id', 'verdict') for P in ['p1', 'p2', 'merged']}
    for i, o in old.items():
        cat = category_t2(i)
        if cat is None:
            continue
        m, a, b = sc['merged'].get(i), sc['p1'].get(i), sc['p2'].get(i)
        if not m:
            continue
        vs = {x['verdict'] for x in (m, a, b) if x}
        rows.append(dict(round='tree_round2', id=i, category=cat, item=o['item'], predicted=o['predicted_range'],
                         old_value=o['measured'], clean_value=m['measured'], ratio_clean_old=ratio_str(o['measured'], m['measured']),
                         old_verdict=o['verdict'], clean_verdict=m['verdict'], p1_verdict=a['verdict'] if a else '',
                         p2_verdict=b['verdict'] if b else '', stable='yes' if len(vs) == 1 else 'UNSTABLE',
                         changed='CHANGED' if o['verdict'] != m['verdict'] else ''))
    old = read_sc(os.path.join(OLD_IC, 'scorecard.csv'), 'id', 'result')
    sc = {P: read_sc(os.path.join(R, 'ic', P, 'scorecard.csv'), 'id', 'result') for P in ['p1', 'p2', 'merged']}
    for i, o in old.items():
        cat = category_ic(i)
        m, a, b = sc['merged'].get(i), sc['p1'].get(i), sc['p2'].get(i)
        if not m:
            continue
        vs = {x['result'] for x in (m, a, b) if x}
        rows.append(dict(round='ic_round', id=i, category=cat, item=o['item'], predicted=o['predicted'],
                         old_value=o['measured'], clean_value=m['measured'], ratio_clean_old=ratio_str(o['measured'], m['measured']),
                         old_verdict=o['result'], clean_verdict=m['result'], p1_verdict=a['result'] if a else '',
                         p2_verdict=b['result'] if b else '', stable='yes' if len(vs) == 1 else 'UNSTABLE',
                         changed='CHANGED' if o['result'] != m['result'] else ''))
    with open(os.path.join(R, 'scorecard.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


# ------------------------------------------------------------------ load-model tests (§3, §5)
def load_old_t2():
    return {r['name']: r for r in csv.DictReader(open(os.path.join(OLD_T2, 'tree_metrics.csv')))}


def ic_rows(path):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r['run'] == '0':
            out[(r['kernel'], r['ic'], int(r['n']), r['leaf'], r['block'], r['mass_spectrum'])] = float(r['ph_traverse_ms'])
    return out


def load_tests(td):
    L = []
    P = L.append
    newm = {r['name']: r for r in csv.DictReader(open(os.path.join(td['merged'], 'tree_metrics.csv')))}
    p1 = {r['name']: r for r in csv.DictReader(open(os.path.join(td['p1'], 'tree_metrics.csv')))}
    p2 = {r['name']: r for r in csv.DictReader(open(os.path.join(td['p2'], 'tree_metrics.csv')))}
    old = load_old_t2()
    r1 = {r['name']: r for r in csv.DictReader(open(os.path.join(OLD_T1, 'tree_metrics.csv')))}
    f = lambda d, k, c='trav': float(d[k][c]) if k in d and d[k].get(c) not in (None, '') else float('nan')
    NS = [65536, 262144, 1048576]
    # test 1: round-1 clean anchors
    P('## T1: clean #4/#5 (leaf 16 b128, set A) vs tree round 1 (0% load)')
    d1 = []
    for k in ['bh_naive', 'bh_sorted']:
        for ic in ['plummer', 'core']:
            for n in NS:
                nm = f'A_{k}_{ic}_{n}'
                a, c = f(r1, nm), f(newm, nm)
                d1.append(c / a - 1)
                P(f'  {nm:34s} round1 {a:9.3f}  clean {c:9.3f}  clean/round1 {c / a:.4f}')
    within = sum(abs(x) <= 0.03 for x in d1)
    med = statistics.median(abs(x) for x in d1)
    P(f'  within +-3%: {within}/12, median |d| {100 * med:.2f}%  -> {"PASS" if within >= 10 and med <= 0.02 else "FAIL"} (need >= 10/12 and <= 2%)')
    # per-kernel clean/old time ratios, round-2 set A
    P('\n## clean/old traversal time, round-2 set A (mean of passes vs old single pass)')
    fam = {}
    for nm in sorted(newm):
        if not nm.startswith('A_') or nm.startswith('A2_') or nm not in old:
            continue
        k = newm[nm]['kernel']
        rr = f(newm, nm) / f(old, nm)
        fam.setdefault(k, []).append(rr)
    for k, v in sorted(fam.items()):
        P(f'  {k:26s} median {statistics.median(v):.4f}  range {min(v):.4f}..{max(v):.4f}  ({len(v)} cells)')
    s5 = fam.get('bh_sorted', [])
    P(f'T2: #5 cells >= 1% faster clean: {sum(x <= 0.99 for x in s5)}/6 -> {"PASS" if sum(x <= 0.99 for x in s5) >= 5 else "FAIL"}')
    w7 = fam.get('bh_warp', [])
    P(f'T4a (round 2): #7 median clean/old {statistics.median(w7):.4f} -> {"PASS" if statistics.median(w7) >= 0.97 else "FAIL"} (need >= 0.97)')
    # speedups
    sp = []
    for ic in ['plummer', 'core']:
        for n in NS:
            a = f(old, f'A_bh_sorted_{ic}_{n}') / f(old, f'A_bh_warp_{ic}_{n}')
            b = f(newm, f'A_bh_sorted_{ic}_{n}') / f(newm, f'A_bh_warp_{ic}_{n}')
            sp.append(b / a)
            P(f'  T5/T7 {ic:8s} {n:8d} old {a:.3f} clean {b:.3f} ratio {b / a:.4f}')
    P(f'T3a (round 2): W8/W9 median clean/old {statistics.median(sp):.4f} -> {"PASS" if statistics.median(sp) <= 1.0 else "FAIL"} (need <= 1.00)')
    # #6 ratios
    sixr = []
    for ic in ['plummer', 'core']:
        for n in NS:
            for arm, sib in [('bh_persistent', 'bh_sorted'), ('bh_persistent_lane', 'bh_sorted'),
                             ('bh_persistent_lane_naive', 'bh_naive'), ('bh_persistent_warp', 'bh_warp')]:
                a = f(old, f'A_{sib}_{ic}_{n}') / f(old, f'A_{arm}_{ic}_{n}')
                b = f(newm, f'A_{sib}_{ic}_{n}') / f(newm, f'A_{arm}_{ic}_{n}')
                sixr.append(b / a)
    # drift test
    dr = []
    for nm in p1:
        if nm in p2 and nm[:2] in ('A_', 'B_', 'C_', 'E_', 'F_', 'G_') and not nm.startswith('A2_'):
            a, b = f(p1, nm, 'trav'), f(p2, nm, 'trav')
            if not (a > 0 and b > 0):
                a, b = f(p1, nm, 'total_ms'), f(p2, nm, 'total_ms')
            if a > 0 and b > 0:
                dr.append(abs(b / a - 1))
    ac = f(newm, 'A_bh_sorted_plummer_1048576') / f(newm, 'C_bh_sorted_n1048576_l16_b128')
    # IC round
    P('\n## IC round: clean/old traversal per kernel (set A best-tuned, set B, set K)')
    icfam, s_ratio = {}, {}
    for sname in ['setA.csv', 'setB.csv', 'setK.csv']:
        o, c = ic_rows(os.path.join(OLD_IC, sname)), ic_rows(os.path.join(R, 'ic', 'merged', sname))
        for k, v in c.items():
            if k in o:
                icfam.setdefault((sname, k[0]), []).append(v / o[k])
    for k, v in sorted(icfam.items()):
        P(f'  {k[0]} {k[1]:14s} median {statistics.median(v):.4f}  range {min(v):.4f}..{max(v):.4f}  ({len(v)} cells)')
    oA, cA = ic_rows(os.path.join(OLD_IC, 'setA.csv')), ic_rows(os.path.join(R, 'ic', 'merged', 'setA.csv'))
    def t(rows, kern, ic, n):
        for k, v in rows.items():
            if k[0] == kern and k[1] == ic and k[2] == n and k[5] == 'equal':
                return v
        return float('nan')
    P('\n## IC S ratios (#7/#5) and Q ratios (T5/T6), clean/old')
    s3d, sce, qr = [], [], []
    for ic in ['plummer', 'disk', 'galaxies', 'cube_shell', 'cube_edges', 'uniform']:
        for n in NS:
            so = t(oA, 'bh_sorted', ic, n) / t(oA, 'bh_warp', ic, n)
            sn = t(cA, 'bh_sorted', ic, n) / t(cA, 'bh_warp', ic, n)
            qo = t(oA, 'bh_sorted', ic, n) / t(oA, 'bh_persistent', ic, n)
            qn = t(cA, 'bh_sorted', ic, n) / t(cA, 'bh_persistent', ic, n)
            (s3d if ic in ('disk', 'galaxies', 'uniform') else sce if ic == 'cube_edges' else []).append(sn / so)
            if ic != 'plummer':
                qr.append(qn / qo)
            P(f'  {ic:10s} {n:8d}  S old {so:.3f} clean {sn:.3f} ({sn / so:.4f}) | Q old {qo:.3f} clean {qn:.3f} ({qn / qo:.4f})')
    oB, cB = ic_rows(os.path.join(OLD_IC, 'setB.csv')), ic_rows(os.path.join(R, 'ic', 'merged', 'setB.csv'))
    pB = cB.get(('bh_sorted', 'plummer', 262144, '16', '128', 'equal'), float('nan'))
    w7B = cB.get(('bh_warp', 'plummer', 1048576, '16', '128', 'equal'), float('nan'))
    ic7 = [v for (s, k), vs in icfam.items() if k == 'bh_warp' and s == 'setA.csv' for v in vs]
    P('\n## Falsification tests (§5)')
    P(f'1. round-1 anchors: {within}/12 within 3%, median |d| {100 * med:.2f}% -> {"PASS" if within >= 10 and med <= 0.02 else "FAIL"}')
    P(f'2a. round-2 #5 faster clean by >= 1%: {sum(x <= 0.99 for x in s5)}/6 -> {"PASS" if sum(x <= 0.99 for x in s5) >= 5 else "FAIL"}')
    P(f'2b. IC #5 plummer 262144 l16 b128: clean {pB:.3f} vs 8.475 ({pB / 8.475 - 1:+.2%}) (old 9.724) -> {"PASS" if abs(pB / 8.475 - 1) <= 0.04 else "FAIL"}')
    P(f'3. median clean/old speedup: round-2 W8/W9 {statistics.median(sp):.4f}; IC 3D S {statistics.median(s3d):.4f} -> '
      f'{"PASS" if statistics.median(sp) <= 1.0 and statistics.median(s3d) <= 1.0 else "FAIL"}')
    P(f'4. #7 median clean/old: round 2 {statistics.median(w7):.4f}, IC set A {statistics.median(ic7):.4f}; IC set B #7 1M l16 b128 clean {w7B:.3f} vs old 6.703 ({w7B / 6.703:.4f}) -> '
      f'{"PASS" if statistics.median(w7) >= 0.97 and statistics.median(ic7) >= 0.97 else "FAIL"}')
    six_all = sixr + qr
    m6 = statistics.median(abs(x - 1) for x in six_all)
    P(f'5. #6 same-family ratios: median |clean/old - 1| {100 * m6:.2f}% over {len(six_all)} cells (round 2 {100 * statistics.median(abs(x - 1) for x in sixr):.2f}%, IC Q {100 * statistics.median(abs(x - 1) for x in qr):.2f}%) -> {"PASS" if m6 <= 0.02 else "FAIL"} (verdict flips: see scorecard)')
    md = statistics.median(dr)
    P(f'6. drift: median |pass2/pass1 - 1| {100 * md:.2f}% over {len(dr)} runs (p90 {100 * sorted(dr)[int(0.9 * len(dr))]:.2f}%); '
      f'#5 1M l16 b128 set A / set C {ac:.4f} -> {"PASS" if md <= 0.02 and abs(ac - 1) <= 0.03 else "FAIL"}')
    ce = max(abs(x - 1) for x in sce)
    P(f'7. H-coh: cube_edges S max |clean/old - 1| {100 * ce:.2f}% (3D median {100 * (1 - statistics.median(s3d)):+.2f}% fall) -> {"PASS" if ce <= 0.03 else "FAIL"}')
    open(os.path.join(R, 'load_tests.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def clocks():
    p = os.path.join(R, 'gpu_clock_log.csv')
    if not os.path.exists(p):
        return
    sm, pw, ut = [], [], []
    for r in csv.reader(open(p)):
        if not r or r[0].startswith('timestamp'):
            continue
        try:
            s = float(r[1].split()[0]); w = float(r[3].split()[0]); u = float(r[6].split()[0])
        except (ValueError, IndexError):
            continue
        if s > 1000:
            sm.append(s); pw.append(w)
        ut.append(u)
    q = lambda v, x: sorted(v)[int(x * (len(v) - 1))]
    s = (f'clock log: {len(ut)} samples; under load ({len(sm)} samples, SM > 1000 MHz): SM median {statistics.median(sm):.0f} MHz '
         f'(p5 {q(sm, 0.05):.0f}), power median {statistics.median(pw):.0f} W, max {max(pw):.0f} W')
    open(os.path.join(R, 'clock_summary.txt'), 'w').write(s + '\n')
    print(s)


if __name__ == '__main__':
    vr = None
    vt = os.path.join(R, 'viewer', 'V2_disk_1M_rate_t3.0.txt')
    if os.path.exists(vt):
        m = re.search(r'([\d.]+) G int/s real', open(vt, errors='replace').read())
        vr = float(m.group(1)) if m else None
    td = tree_passes()
    ic_passes(vr)
    rows = build_scorecard(td, None)
    clocks()
    load_tests(td)
    for cat in ['primary', 'secondary', 'identity']:
        rr = [r for r in rows if r['category'] == cat]
        print(f'{cat}: {len(rr)} rows, old HIT {sum(r["old_verdict"] == "HIT" for r in rr)}, clean HIT {sum(r["clean_verdict"] == "HIT" for r in rr)}, '
              f'changed {[r["id"] for r in rr if r["changed"]]}, unstable {[r["id"] for r in rr if r["stable"] != "yes"]}')
