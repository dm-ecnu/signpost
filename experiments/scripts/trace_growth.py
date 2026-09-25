#!/usr/bin/env python3
"""Growth of distinct requested objects over each efficiency trace (Discussion,
'When eager construction pays off'). Reads results/efficiency/*.trace.json and
*.manifest.json; no model, ES, or network calls."""
import glob, json
from pathlib import Path

res = Path(__file__).resolve().parents[1] / 'results/efficiency'
for f in sorted(glob.glob(str(res / '*.trace.json'))):
    d = Path(f).name.split('.')[0]
    n = json.loads((res / f'{d}.manifest.json').read_text())['objects']
    tr = json.loads(Path(f).read_text())['trace']
    seen, cum = set(), []
    for q in tr:
        seen |= set(q['objects'])
        cum.append(len(seen))
    h = len(cum) // 2
    first, second = cum[h - 1] / h, (cum[-1] - cum[h - 1]) / (len(cum) - h)
    print(f'{d:32s} objects={n} questions={len(tr)} touched={cum[-1]} ({100 * cum[-1] / n:.2f}%) '
          f'new/q first half={first:.2f} second half={second:.2f} questions to build all >= {n / second:,.0f}')
