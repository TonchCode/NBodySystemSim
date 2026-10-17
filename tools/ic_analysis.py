#!/usr/bin/env python3
"""IC-round analysis: derived tables + the committed scorecard (docs/predictions/2026-10-17-ic-round.md).

    python tools/ic_analysis.py results/2026-10-17_ic_round [--viewer-rate 1M_disk_G_per_s]

Reads setA.csv (best-tuned grid), setB.csv (leaf 16 b128), setK.csv (mass spectrum), logs/*.txt (light-body
p99 of the heavy/light split, X_* stack-change timings) and timeline_metrics.csv (written by
tools/timeline_analysis.py). Writes derived.csv and scorecard.csv. Stdlib only.
"""
import csv, os, re, statistics, sys

d = sys.argv[1]
viewer_rate = None
if '--viewer-rate' in sys.argv:
    viewer_rate = float(sys.argv[sys.argv.index('--viewer-rate') + 1])

def load(name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    out, seen = [], set()
    for r in rows:   # one row per invocation (run 0 carries the medians; per-run ms kept separately)
        if r['run'] == '0':
            out.append(r)
    # attach per-run ms
    for r in out:
        r['_ms'] = [float(x['ms']) for x in rows if x['kernel'] == r['kernel'] and x['n'] == r['n'] and x['ic'] == r['ic']
                    and x['leaf'] == r['leaf'] and x['block'] == r['block'] and x['mass_spectrum'] == r['mass_spectrum']]
    return out

A, B, K = load('setA.csv'), load('setB.csv'), load('setK.csv')
NS = [65536, 262144, 1048576]
NEW = ['disk', 'galaxies', 'cube_shell', 'cube_edges', 'uniform']
ALL = ['plummer'] + NEW
f = float

def get(rows, kernel, ic, n, spec='equal'):
    for r in rows:
        if r['kernel'] == kernel and r['ic'] == ic and int(r['n']) == n and r['mass_spectrum'] == spec:
            return r
    return None

def trav(r):
    return f(r['ph_traverse_ms'])

out = []
def P(s=''):
    print(s); out.append(s)

# ---- derived table
der = []
P('== set A (best-tuned, theta 0.5), traversal ms (median of 11), work, imbalance, error  [BACKGROUND-LOAD LABEL]')
P('%-10s %8s | %8s %8s %8s %8s | %6s %6s %6s | %6s %6s | %6s %6s | %7s %7s | %5s %5s %6s' % (
    'ic', 'n', 'T#4', 'T#5', 'T#6', 'T#7', 'I#4', 'I#5', 'I#7', 'wmm4', 'wmm5', 'wc5', 'wc7', '#7/#5', '#5/#6', 'push', 'depth', 'dup%'))
for ic in ALL:
    for n in NS:
        r4, r5, r6, r7 = (get(A, k, ic, n) for k in ('bh_naive', 'bh_sorted', 'bh_persistent', 'bh_warp'))
        if not all((r4, r5, r6, r7)):
            continue
        row = dict(ic=ic, n=n, t4=trav(r4), t5=trav(r5), t6=trav(r6), t7=trav(r7),
                   i4=f(r4['inter_per_body']), i5=f(r5['inter_per_body']), i7=f(r7['inter_per_body']),
                   wmm4=f(r4['warp_max_mean']), wmm5=f(r5['warp_max_mean']), wmm7=f(r7['warp_max_mean']),
                   wc5=f(r5['warp_cost_max_mean']), wc7=f(r7['warp_cost_max_mean']),
                   simt7i=f(r7['simt_eff_inter']), simt7v=f(r7['simt_eff_visit']),
                   s75=trav(r5) / trav(r7), q=trav(r5) / trav(r6), s54=trav(r4) / trav(r5),
                   push=int(r5['push_depth']), depth=int(r5['tree_depth']), sb=int(r5['stack_bound']),
                   dup=int(r5['dup_keys']) / n * 100, ov=sum(int(x['stack_overflows']) for x in (r4, r5, r6, r7)),
                   n5rms=f(r5['norm_rms']), n7rms=f(r7['norm_rms']), n5p99=f(r5['norm_p99']), n7p99=f(r7['norm_p99']),
                   rate5=f(r5['trav_ginter_per_s']), rate7=f(r7['trav_ginter_per_s']),
                   tot5=f(r5['ms']) if False else statistics.median(r5['_ms']), tot7=statistics.median(r7['_ms']),
                   g1_5=r5['g1'], g1_7=r7['g1'], gates=','.join(x['gates'] for x in (r4, r5, r6, r7)),
                   exact6=r6['exact_match'] + '/' + r6['counts_match'])
        der.append(row)
        P('%-10s %8d | %8.3f %8.3f %8.3f %8.3f | %6.0f %6.0f %6.0f | %6.3f %6.3f | %6.2f %6.2f | %7.2f %7.3f | %5d %5d %6.3f' % (
            ic, n, row['t4'], row['t5'], row['t6'], row['t7'], row['i4'], row['i5'], row['i7'], row['wmm4'], row['wmm5'],
            row['wc5'], row['wc7'], row['s75'], row['q'], row['push'], row['depth'], row['dup']))
P()
P('rates (traversal G int/s) and errors (normalised rms #5/#7; legacy TG1 #5/#7; gates #4,#5,#6,#7; #6 exact/counts)')
for r in der:
    P('%-10s %8d  rate #5 %6.1f #7 %6.1f | #4/#5 %.2f | norm rms %.2e / %.2e p99 %.2e / %.2e | TG1 %s/%s | %s | %s | full-step #7/#5 %.2f' % (
        r['ic'], r['n'], r['rate5'], r['rate7'], r['s54'], r['n5rms'], r['n7rms'], r['n5p99'], r['n7p99'], r['g1_5'], r['g1_7'],
        r['gates'], r['exact6'], r['tot5'] / r['tot7']))
with open(os.path.join(d, 'derived.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(der[0].keys()))
    w.writeheader(); w.writerows(der)
D = {(r['ic'], r['n']): r for r in der}

P()
P('== set B (leaf 16 b128): #5 / #7 traversal, work, union ratio, normalised rms')
for ic in ALL:
    for n in NS:
        r5, r7 = get(B, 'bh_sorted', ic, n), get(B, 'bh_warp', ic, n)
        if r5 and r7:
            P('%-10s %8d  T5 %8.3f T7 %7.3f (x%.2f) | I5 %6.0f I7 %6.0f union x%.3f | norm rms %.3e / %.3e | push %s dup %s' % (
                ic, n, trav(r5), trav(r7), trav(r5) / trav(r7), f(r5['inter_per_body']), f(r7['inter_per_body']),
                f(r7['union_inter_ratio']), f(r5['norm_rms']), f(r7['norm_rms']), r5['push_depth'], r5['dup_keys']))
P()
P('== set K (262144, leaf 16 b128): mass spectrum')
for spec in [('plummer', 'salpeter'), ('plummer', 'heavy-few'), ('disk', 'salpeter')]:
    for k in ('bh_sorted', 'bh_warp'):
        r = get(K, k, spec[0], 262144, spec[1])
        if r:
            P('%-8s %-9s %-9s T %7.3f I %6.0f norm rms %.3e p99 %.3e heavy max %s mass max/min %s gates %s' % (
                spec[0], spec[1], k, trav(r), f(r['inter_per_body']), f(r['norm_rms']), f(r['norm_p99']), r['heavy_norm_max'],
                r['mass_max_min'], r['gates']))

# light-body p99 from the logs (K5)
def light_p99(log):
    p = os.path.join(d, 'logs', log + '.txt')
    if not os.path.exists(p):
        return None, None
    for line in open(p, errors='replace'):
        m = re.search(r'heavy \(m >= 10 x median, (\d+) bodies\) rms (\S+) max (\S+) \| light \((\d+)\) rms (\S+) p99 (\S+) max (\S+)', line)
        if m:
            return float(m.group(3)), float(m.group(6))
    return None, None

# stack-change check (X)
P()
P('== X: pre-round binary vs this binary (kStack 64 -> 96), 262144, leaf 16 b128, traversal ms')
xs = {}
for fn in sorted(os.listdir(os.path.join(d, 'logs'))):
    m = re.match(r'X_(old|new)_(bh_\w+?)_(plummer|core)_r(\d)\.txt', fn)
    if not m:
        continue
    for line in open(os.path.join(d, 'logs', fn), errors='replace'):
        mm = re.search(r'traverse (\d+\.\d+)\s+sum', line)
        if mm:
            xs.setdefault((m.group(2), m.group(3)), {}).setdefault(m.group(1), []).append(float(mm.group(1)))
for k, v in sorted(xs.items()):
    if 'old' in v and 'new' in v:
        P('%-10s %-8s old %s new %s  new/old %.4f' % (k[0], k[1], v['old'], v['new'], statistics.mean(v['new']) / statistics.mean(v['old'])))

# timelines
tl = {}
tp = os.path.join(d, 'timeline_metrics.csv')
if os.path.exists(tp):
    for r in csv.DictReader(open(tp)):
        tl[os.path.basename(r.get('file', ''))] = r
    P(); P('== timelines (galaxies 65536, warp units): ' + '; '.join('%s sm_eff %s tail %s' % (k, v.get('sm_eff'), v.get('tail_frac', v.get('tail'))) for k, v in tl.items()))

# ---------------------------------------------------------------- scorecard
sc = []
def rng(i, desc, val, lo, hi, anchored=False, src='', note=''):
    hit = val is not None and lo <= val <= hi
    sc.append(dict(id=i, item=desc, predicted='%g-%g' % (lo, hi), measured='na' if val is None else '%.4g' % val,
                   result='HIT' if hit else 'MISS', anchored='A' if anchored else '', source=src, note=note))
def dirn(i, desc, cond, measured, anchored=False, src='', note='', construction=False):
    sc.append(dict(id=i, item=desc, predicted='holds' + (' (C)' if construction else ''), measured=measured,
                   result='HIT' if cond else 'MISS', anchored='A' if anchored else '', source=src, note=note))

I_PRED = {
    'disk':       [(520, 610, 670, 970, 930, 1280), (600, 820, 780, 1320, 1070, 1730), (840, 1710, 840, 1710, 870, 1910)],
    'galaxies':   [(470, 550, 610, 870, 860, 1170), (540, 770, 700, 1220, 990, 1640), (750, 1590, 750, 1590, 800, 1820)],
    'cube_shell': [(136, 160, 180, 260, 270, 380), (129, 176, 170, 280, 260, 420), (160, 310, 160, 310, 180, 390)],
    'cube_edges': [(52, 60, 67, 97, 82, 115), (49, 67, 64, 106, 78, 126), (60, 117, 60, 117, 56, 118)],
    'uniform':    [(340, 400, 440, 640, 750, 1040), (370, 500, 480, 800, 810, 1300), (510, 1000, 510, 1000, 650, 1380)],
}
idn = {'disk': 1, 'galaxies': 2, 'cube_shell': 3, 'cube_edges': 4, 'uniform': 5}
for ic, cells in I_PRED.items():
    for j, n in enumerate(NS):
        r = D.get((ic, n))
        for kk, (lab, key) in enumerate((('4', 'i4'), ('5', 'i5'), ('7', 'i7'))):
            lo, hi = cells[j][2 * kk], cells[j][2 * kk + 1]
            rng('I%d-%s-%d' % (idn[ic], lab, n), '%s #%s int/body N=%d (best-tuned leaf)' % (ic, lab, n), r[key] if r else None,
                lo, hi, anchored=(n == 65536), src='setA.csv')
order = ['plummer', 'disk', 'galaxies', 'uniform', 'cube_shell', 'cube_edges']
ok = all(all(D[(order[k], n)]['i5'] > D[(order[k + 1], n)]['i5'] for k in range(5)) for n in NS)
dirn('I6', '#5 work order plummer > disk > galaxies > uniform > cube_shell > cube_edges at every N', ok,
     '; '.join('%d: ' % n + ' '.join('%s %.0f' % (o, D[(o, n)]['i5']) for o in order) for n in NS), src='setA.csv')
def bI7(ic, n):
    r = get(B, 'bh_warp', ic, n); return f(r['inter_per_body']) if r else None
vals = {ic: bI7(ic, 1048576) / bI7(ic, 65536) for ic in ('cube_shell', 'cube_edges', 'plummer')}
dirn('I7', 'sheet/line #7 work flat (1M/65536 within 0.85-1.15), plummer grows >= 10% (leaf 16)',
     0.85 <= vals['cube_shell'] <= 1.15 and 0.85 <= vals['cube_edges'] <= 1.15 and vals['plummer'] >= 1.10,
     ' '.join('%s %.3f' % kv for kv in vals.items()), src='setB.csv')

M_PRED = {'disk': (1.35, 1.80, 1.06, 1.15), 'galaxies': (1.35, 1.80, 1.06, 1.15), 'cube_shell': (1.25, 1.70, 1.05, 1.20),
          'cube_edges': (1.40, 2.30, 1.05, 1.30), 'uniform': (1.08, 1.25, 1.02, 1.10)}
for ic, (a, b, c, e) in M_PRED.items():
    for n in NS:
        r = D[(ic, n)]
        rng('M%d-4-%d' % (idn[ic], n), '%s #4 per-warp max/mean N=%d' % (ic, n), r['wmm4'], a, b, src='setA.csv')
        rng('M%d-5-%d' % (idn[ic], n), '%s #5 per-warp max/mean N=%d' % (ic, n), r['wmm5'], c, e, src='setA.csv')
w = {ic: D[(ic, 262144)]['wmm4'] for ic in ALL}
cond = w['cube_edges'] > max(w['disk'], w['galaxies']) and min(w['disk'], w['galaxies']) > w['cube_shell'] > w['plummer'] > w['uniform']
dirn('M6', '#4 max/mean order cube_edges > disk,galaxies > cube_shell > plummer > uniform at 262144', cond,
     ' '.join('%s %.3f' % kv for kv in w.items()), src='setA.csv')
cond = all(abs(r['wmm7'] - 1) < 1e-9 and r['simt7i'] > 0.9999 and r['simt7v'] > 0.9999 for r in der)
dirn('M7', '#7 per-warp max/mean = 1 and SIMT eff 1.000 at both sites, every IC and N', cond,
     'min simt %.4f' % min(min(r['simt7i'], r['simt7v']) for r in der), src='setA.csv', construction=True)

W_PRED = {'galaxies': (1, 1.6, 2.2), 'disk': (2, 1.5, 2.1), 'cube_edges': (3, 1.3, 2.0), 'cube_shell': (4, 1.2, 1.6), 'uniform': (5, 1.1, 1.5)}
for ic, (k, a, b) in W_PRED.items():
    for n in NS:
        rng('W%d-%d' % (k, n), '%s #5 warp-cost max/mean N=%d' % (ic, n), D[(ic, n)]['wc5'], a, b, src='setA.csv')
wc = {ic: D[(ic, 65536)]['wc5'] for ic in NEW}
dirn('W6', 'galaxies has the largest #5 warp-cost max/mean of the new ICs at 65536', max(wc, key=wc.get) == 'galaxies',
     ' '.join('%s %.2f' % kv for kv in wc.items()), src='setA.csv')

S_PRED = {'disk': [(2.5, 4.5), (4.0, 7.0), (4.5, 8.0)], 'galaxies': [(2.5, 4.5), (4.0, 7.0), (4.5, 8.0)],
          'cube_shell': [(1.5, 3.5), (2.0, 5.0), (2.0, 6.0)], 'cube_edges': [(1.0, 3.0), (1.2, 4.0), (1.5, 5.0)],
          'uniform': [(1.8, 3.5), (3.0, 5.5), (3.5, 6.5)]}
for ic, cells in S_PRED.items():
    for j, n in enumerate(NS):
        rng('S%d-%d' % (idn[ic], n), '%s #7/#5 traversal speedup (best vs best) N=%d' % (ic, n), D[(ic, n)]['s75'], *cells[j],
            src='setA.csv')
cond = all(D[(ic, n)]['s75'] > 1 for ic in ALL for n in (262144, 1048576))
dirn('S6', '#7 faster than #5 on every IC at N >= 262144', cond,
     'min %.2f' % min(D[(ic, n)]['s75'] for ic in ALL for n in (262144, 1048576)), src='setA.csv')
s1 = {ic: D[(ic, 1048576)]['s75'] for ic in NEW}
cond = max(s1, key=s1.get) in ('disk', 'galaxies') and min(s1, key=s1.get) == 'cube_edges'
dirn('S7', 'at 1M the #7/#5 speedup is highest on disk/galaxies, lowest on cube_edges (new ICs)', cond,
     ' '.join('%s %.2f' % kv for kv in s1.items()), src='setA.csv')

rng('Q1', 'galaxies 65536 T5/T6 (warp queue)', D[('galaxies', 65536)]['q'], 0.85, 0.99, src='setA.csv', note='load-sensitive')
rng('Q2', 'galaxies 262144 T5/T6', D[('galaxies', 262144)]['q'], 0.86, 0.97, src='setA.csv', note='load-sensitive')
rng('Q3', 'galaxies 1M T5/T6', D[('galaxies', 1048576)]['q'], 0.86, 0.97, src='setA.csv', note='load-sensitive')
for ic in ('disk', 'cube_shell', 'cube_edges', 'uniform'):
    for n in NS:
        rng('Q4-%s-%d' % (ic, n), '%s %d T5/T6' % (ic, n), D[(ic, n)]['q'], 0.84, 0.99, src='setA.csv', note='load-sensitive')
qa = {(ic, n): D[(ic, n)]['q'] for ic in ALL for n in NS}
best = max(qa, key=qa.get)
dirn('Q5', 'no IC where #6 beats #5 (all T5/T6 <= 1.00)', all(v <= 1.0 for v in qa.values()),
     'max %.3f (%s %d)' % (qa[best], best[0], best[1]), src='setA.csv', note='load-sensitive')
k5, k6 = 'T_bh_sorted_galaxies_65536.csv', 'T_bh_persistent_galaxies_65536.csv'
if k5 in tl and k6 in tl:
    e5, e6 = float(tl[k5]['sm_eff']), float(tl[k6]['sm_eff'])
    dirn('Q6', 'galaxies 65536: #6 SM efficiency >= #5 (timelines) while Q1 shows no speed gain',
         e6 >= e5 and D[('galaxies', 65536)]['q'] <= 1.0, 'sm_eff #5 %.3f #6 %.3f, T5/T6 %.3f' % (e5, e6, D[('galaxies', 65536)]['q']),
         src='timeline_metrics.csv')
else:
    dirn('Q6', 'timelines', False, 'missing', src='timeline_metrics.csv')

E0 = {'disk': (5.56e-3, 3.93e-3), 'galaxies': (6.08e-3, 4.06e-3), 'cube_shell': (6.72e-3, 5.48e-3),
      'cube_edges': (9.54e-3, 8.11e-3), 'uniform': (3.20e-3, 2.13e-3)}
for ic, (b5, b7) in E0.items():
    for n in (262144, 1048576):
        r5, r7 = get(B, 'bh_sorted', ic, n), get(B, 'bh_warp', ic, n)
        rng('E1-%s-%d' % (ic, n), '%s #5 norm rms / 65536 value, N=%d (leaf 16)' % (ic, n), f(r5['norm_rms']) / b5, 0.85, 1.25, src='setB.csv')
        rng('E2-%s-%d' % (ic, n), '%s #7 norm rms / 65536 value, N=%d (leaf 16)' % (ic, n), f(r7['norm_rms']) / b7, 0.85, 1.25, src='setB.csv')
allrows = A + B + K
tn = [r for r in allrows if r['tn1'] != 'PASS' or r['tn2'] != 'PASS']
dirn('E3', 'TN1 and TN2 PASS in every run (sets A, B, K)', not tn, '%d of %d runs fail' % (len(tn), len(allrows)), src='setA/B/K')
t3 = [r for r in allrows if r['kernel'] == 'bh_warp' and r['tg3n'] != 'PASS']
dirn('E4', 'TG3n PASS for #7 vs #5, every IC and N', not t3, '%d of %d bh_warp runs fail' % (len(t3), sum(r['kernel'] == 'bh_warp' for r in allrows)), src='setA/B/K')
def g1(ic, k):
    return [r['g1'] for r in A + B if r['ic'] == ic and r['kernel'] == k and r['mass_spectrum'] == 'equal']
c1 = all(x == 'FAIL' for x in g1('cube_edges', 'bh_sorted') + g1('cube_edges', 'bh_warp') + g1('cube_shell', 'bh_sorted'))
c2 = all(x == 'PASS' for ic in ('disk', 'galaxies', 'uniform') for k in ('bh_sorted', 'bh_warp') for x in g1(ic, k))
dirn('E5', 'legacy TG1 FAIL: #5,#7 cube_edges and #5 cube_shell (all N); PASS on disk, galaxies, uniform', c1 and c2,
     'edges #5 %s #7 %s | shell #5 %s | disk/gal/uni all PASS: %s' % (g1('cube_edges', 'bh_sorted'), g1('cube_edges', 'bh_warp'),
                                                                     g1('cube_shell', 'bh_sorted'), c2), src='setA/B')
fp = [f(r['fp64_norm_max']) for r in A + B + K if r['kernel'] == 'bh_sorted']
dirn('E6', '#5 FP64-sample normMax <= 0.05 on every run', max(fp) <= 0.05, 'max %.4f' % max(fp), src='setA/B/K')

def kr(k, ic, spec):
    return get(K, k, ic, 262144, spec)
pe5, pe7 = get(B, 'bh_sorted', 'plummer', 262144), get(B, 'bh_warp', 'plummer', 262144)
de5 = get(B, 'bh_sorted', 'disk', 262144)
rng('K1', '#5 work plummer salpeter / equal (262144)', f(kr('bh_sorted', 'plummer', 'salpeter')['inter_per_body']) / f(pe5['inter_per_body']),
    1.08, 1.22, anchored=True, src='setK/setB')
rng('K2', '#5 work plummer heavy-few / equal', f(kr('bh_sorted', 'plummer', 'heavy-few')['inter_per_body']) / f(pe5['inter_per_body']),
    0.98, 1.04, anchored=True, src='setK/setB')
sp_s = trav(kr('bh_sorted', 'plummer', 'salpeter')) / trav(kr('bh_warp', 'plummer', 'salpeter'))
sp_e = trav(pe5) / trav(pe7)
rng('K3', '#7/#5 speedup salpeter / equal plummer', sp_s / sp_e, 0.88, 1.12, src='setK/setB', note='load-sensitive')
for spec in ('salpeter', 'heavy-few'):
    for k, ref in (('bh_sorted', pe5), ('bh_warp', pe7)):
        rng('K4-%s-%s' % (spec, k), 'norm rms %s / equal plummer, %s' % (spec, k), f(kr(k, 'plummer', spec)['norm_rms']) / f(ref['norm_rms']),
            0.85, 1.20, src='setK/setB')
k5rows = []
for log in ('K_bh_sorted_plummer_salpeter', 'K_bh_warp_plummer_salpeter', 'K_bh_sorted_plummer_heavy-few', 'K_bh_warp_plummer_heavy-few',
            'B_bh_sorted_disk_262144', 'B_bh_warp_disk_262144'):
    hm, lp = light_p99(log)
    k5rows.append((log, hm, lp))
cond = all(hm is not None and hm <= 2 * lp for _, hm, lp in k5rows)
dirn('K5', 'heavy bodies normalised max <= 2 x light p99 (salpeter, heavy-few, disk)', cond,
     '; '.join('%s %s/%s' % (l.replace('K_', '').replace('B_', ''), hm, lp) for l, hm, lp in k5rows), src='logs')
rng('K6', 'disk salpeter / equal #5 work (262144)', f(kr('bh_sorted', 'disk', 'salpeter')['inter_per_body']) / f(de5['inter_per_body']),
    1.05, 1.25, src='setK/setB')

M1 = {ic: D[(ic, 1048576)] for ic in NEW}
dirn('D1', 'push depth <= 45, stack bound <= 47, 0 overflows at 1M (new ICs, all kernels)',
     all(r['push'] <= 45 and r['sb'] <= 47 and r['ov'] == 0 for r in M1.values()),
     ' '.join('%s %d/%d/%d' % (ic, r['push'], r['sb'], r['ov']) for ic, r in M1.items()), src='setA.csv')
rng('D2', 'cube_edges dup keys at 1M (% of bodies)', M1['cube_edges']['dup'], 1.5, 2.6, src='setA.csv')
dirn('D3', 'cube_shell dup keys at 1M < 0.01%', M1['cube_shell']['dup'] < 0.01, '%.4f%%' % M1['cube_shell']['dup'], src='setA.csv')
dep = {ic: r['depth'] for ic, r in M1.items()}
dirn('D4', 'max depth at 1M: cube_edges 42-60 and deepest new IC; cube_shell <= 34; uniform <= 30',
     42 <= dep['cube_edges'] <= 60 and max(dep, key=dep.get) == 'cube_edges' and dep['cube_shell'] <= 34 and dep['uniform'] <= 30,
     ' '.join('%s %d' % kv for kv in dep.items()), src='setA.csv')
dirn('D5', 'disk and galaxies push depth at 1M in 26-36', all(26 <= M1[ic]['push'] <= 36 for ic in ('disk', 'galaxies')),
     'disk %d galaxies %d' % (M1['disk']['push'], M1['galaxies']['push']), src='setA.csv')

shell = [r for r in A + B if r['ic'] == 'cube_shell']
dirn('C1', 'cube_shell does not break the tree: all gates PASS, 0 overflows, dup < 0.01% (and D3)',
     all(r['gates'] == 'PASS' and r['stack_overflows'] == '0' for r in shell) and M1['cube_shell']['dup'] < 0.01,
     '%d runs: gates %s' % (len(shell), sorted(set(r['gates'] for r in shell))), src='setA/B')
dirn('C2', 'cube_shell #5 legacy TG1 FAIL (all N)', all(x == 'FAIL' for x in g1('cube_shell', 'bh_sorted')), str(g1('cube_shell', 'bh_sorted')), src='setA/B')
un = {(ic, n): f(get(B, 'bh_warp', ic, n)['union_inter_ratio']) for ic in ALL for n in NS}
cond = all(2.0 <= un[('cube_shell', n)] <= 2.4 and all(un[(ic, n)] < un[('cube_shell', n)] for ic in ALL if ic not in ('cube_shell', 'uniform'))
           for n in NS)
dirn('C3', 'cube_shell union ratio 2.0-2.4 at every N, second only to uniform', cond,
     '; '.join('%d: ' % n + ' '.join('%s %.2f' % (ic, un[(ic, n)]) for ic in ALL) for n in NS), src='setB.csv')
cond = (max(dep, key=dep.get) == 'cube_edges' and D[('cube_edges', 262144)]['wmm4'] == max(D[(ic, 262144)]['wmm4'] for ic in ALL)
        and all(x == 'FAIL' for x in g1('cube_edges', 'bh_sorted')) and M1['cube_edges']['dup'] >= 1.0)
dirn('C4', 'cube_edges = the stress case: deepest new tree, largest #4 max/mean (262144), legacy TG1 FAIL, dup >= 1%', cond,
     'depth %d, wmm4 %.3f, dup %.2f%%' % (dep['cube_edges'], D[('cube_edges', 262144)]['wmm4'], M1['cube_edges']['dup']), src='setA.csv')
dirn('V1', 'viewer live N / IC / spectrum switching without errors', True, 'phase1 smoke_livekeys.txt + phase-2 presentation runs',
     src='screens/, presentation/', construction=True)
if viewer_rate is not None:
    rng('V2', 'viewer bh_warp disk 1M real rate (G int/s, incl. build + integrator)', viewer_rate, 150, 350, src='presentation/17_disk_1M_rate_t3.0.txt', note='disputed reading: title at t = 3 (evolved, clumpy disk: 2857 int/body vs 1619 at t = 0); not measured at t = 0')

with open(os.path.join(d, 'scorecard.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=['id', 'item', 'predicted', 'measured', 'result', 'anchored', 'source', 'note'])
    w.writeheader(); w.writerows(sc)
hits = sum(r['result'] == 'HIT' for r in sc)
P()
P('== scorecard: %d HIT / %d MISS of %d rows' % (hits, len(sc) - hits, len(sc)))
for prefix in ('I', 'M', 'W', 'S', 'Q', 'E', 'K', 'D', 'C', 'V'):
    rows = [r for r in sc if r['id'].startswith(prefix)]
    P('  %s: %d/%d HIT; MISS: %s' % (prefix, sum(r['result'] == 'HIT' for r in rows), len(rows),
                                     ', '.join('%s (%s vs %s)' % (r['id'], r['measured'], r['predicted']) for r in rows if r['result'] == 'MISS')))
open(os.path.join(d, 'analysis_output.txt'), 'w').write('\n'.join(out) + '\n')
