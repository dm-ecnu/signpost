#!/usr/bin/env python3
"""X4 fit: predict t/t_full of dependency-set maintenance from features a store
knows without recomputing views. Fit on the five frozen e4 corpora; evaluate by
leave-one-corpus-out and, when available, on the held-out MuSiQue row (X2).

Usage: x4_fit.py <x4 feature dir> [<musique e4 json>] > report.json
"""
import json, sys
from pathlib import Path

FIT = ['mix', 'graphrag-bench-medical', 'graphrag-bench-novel', 'agriculture', 'legal']
MODELS = {  # name -> (feature, has_intercept)
    'dirty_share_scaled': ('dirty_share', False),
    'bytes_share_identity': ('bytes_share', None),   # no parameters: t_hat = bytes share
    'bytes_share_scaled': ('bytes_share', False),
    'bytes_share_affine': ('bytes_share', True),
    'entries_share_identity': ('entries_share', None),
    'entries_share_scaled': ('entries_share', False),
    'entries_share_affine': ('entries_share', True),
    'degree_share_identity': ('degree_share', None),
    'degree_share_scaled': ('degree_share', False),
    'degree_share_affine': ('degree_share', True),
}


def fit(xs, ys, intercept):
    if intercept is None:
        return 1.0, 0.0
    if not intercept:
        return sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs), 0.0
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return a, my - a * mx


def cells(d, name, e4=None):
    rep = json.loads((d / f'x4_{name}.json').read_text())
    rows = rep['batches']
    if e4 is not None:
        by = {e['edit_fraction']: e for e in e4['edits'] if e['granularity'] == 'chunk'}
        for r in rows:
            assert by[r['edit_fraction']]['dirty_objects'] == r['dirty_objects']
            r['measured_share_of_full'] = by[r['edit_fraction']]['share_of_full']
    return [dict(r, corpus=name) for r in rows]


def evaluate(train, test, feat, ic):
    a, c = fit([r[feat] for r in train], [r['measured_share_of_full'] for r in train], ic)
    out = []
    for r in test:
        p = a * r[feat] + c
        y = r['measured_share_of_full']
        out.append({'corpus': r['corpus'], 'edit_fraction': r['edit_fraction'], 'pred': p, 'measured': y,
                    'abs_err': abs(p - y), 'decision_ok': (p < 1) == (y < 1)})
    return (a, c), out


def summarize(rows):
    return {'mae': sum(r['abs_err'] for r in rows) / len(rows), 'max_err': max(r['abs_err'] for r in rows),
            'decisions_ok': sum(r['decision_ok'] for r in rows), 'cells': len(rows)}


def main():
    d = Path(sys.argv[1])
    data = {n: cells(d, n) for n in FIT}
    held = None
    if len(sys.argv) > 2:
        held = cells(d, 'musique', json.loads(Path(sys.argv[2]).read_text()))
    allfit = [r for n in FIT for r in data[n]]
    report = {'fit_corpora': FIT, 'models': {}}
    for m, (feat, ic) in MODELS.items():
        params, insample = evaluate(allfit, allfit, feat, ic)
        loco = []
        for n in FIT:
            _, o = evaluate([r for k in FIT if k != n for r in data[k]], data[n], feat, ic)
            loco += o
        entry = {'feature': feat, 'params': {'a': params[0], 'c': params[1]}, 'in_sample': summarize(insample),
                 'loco': summarize(loco), 'loco_cells': loco}
        if held:
            _, o = evaluate(allfit, held, feat, ic)
            entry['musique_holdout'] = summarize(o)
            entry['musique_cells'] = o
        report['models'][m] = entry
    json.dump(report, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
