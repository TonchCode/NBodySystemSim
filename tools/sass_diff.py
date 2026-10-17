#!/usr/bin/env python3
"""Check that every kernel in a BEFORE cuobjdump -sass dump has an instruction-for-instruction
identical body somewhere in the AFTER dump (names may differ, e.g. a new template flag).
Used to show the production kernels are unchanged by the round-2 instrumentation.

    python tools/sass_diff.py before.sass after.sass
Comparison text = the instruction field of every SASS line (addresses and encodings dropped).
"""
import re,sys,glob,os,hashlib
def funcs(path):
    out={}; name=None; body=[]
    for line in open(path,errors='replace'):
        m=re.search(r'Function : (\S+)',line)
        if m:
            if name: out[name]=body
            name=m.group(1); body=[]; continue
        m=re.search(r'/\*[0-9a-f]{4,}\*/\s+(.*?);',line)
        if m and name: body.append(re.sub(r'\s+',' ',m.group(1)))
    if name: out[name]=body
    return out
if len(sys.argv) != 3: sys.exit('usage: sass_diff.py before.sass after.sass')
if True:
    f = sys.argv[2]
    b=funcs(sys.argv[1]); a=funcs(sys.argv[2])
    ah={}
    for n,body in a.items(): ah.setdefault(hashlib.md5('\n'.join(body).encode()).hexdigest(),[]).append(n)
    ok=0; bad=[]
    for n,body in b.items():
        h=hashlib.md5('\n'.join(body).encode()).hexdigest()
        if h in ah: ok+=1
        else: bad.append(n)
    print(f, 'before',len(b),'after',len(a),'identical-matched',ok,'unmatched',len(bad))
    for n in bad[:5]: print('  ',n)

