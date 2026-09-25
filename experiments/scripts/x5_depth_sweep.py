#!/usr/bin/env python3
"""X5: depth sweep of navigation views, b in {1,2,4,8,16,32,64,inf}.

For every served object (chunk/summary/entity nodes, semantic relation edges)
build sigma(o) with the project's own builders, cap it with the same
_apply_cue_topb_item used at serving time, and record per budget:
  entries  per-object entry count (list lengths after the cap, plus the
           size-<=1 prev/next and chunk locate fields), as in
           recount_cue_exposure.py
  bytes    serialized size of the capped view (canonical JSON)
Aggregates: total entries and bytes per budget, per-type mean/P95/max entries.
No model, ES, or network calls.
"""
import argparse, copy, hashlib, json, math, platform, sys, time
from collections import defaultdict
from pathlib import Path

BUDGETS = [1, 2, 4, 8, 16, 32, 64, None]
FIELDS = {
    'chunk': [('vertical', 'parent_summaries')],
    'summary': [('vertical', 'child_summaries'), ('vertical', 'child_chunks'),
                ('provenance', 'source_chunk_ids'), ('provenance', 'source_locates')],
    'entity': [('semantic', 'neighboring_entities'), ('provenance', 'source_chunk_ids'),
               ('provenance', 'source_locates')],
    'relation': [('semantic', 'neighboring_entities'), ('provenance', 'source_chunk_ids'),
                 ('provenance', 'source_locates')],
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def encode(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def entries(v, rtype):
    n = 0
    h = v.get('horizontal')
    if isinstance(h, dict):
        n += sum(1 for k in ('previous_chunk', 'next_chunk') if h.get(k))
    p = v.get('provenance')
    if isinstance(p, dict) and p.get('locate'):
        n += 1
    for sec, field in FIELDS[rtype]:
        s = v.get(sec)
        if isinstance(s, dict) and isinstance(s.get(field), list):
            n += len(s[field])
    return n


def pct(vals, q):
    k = (len(vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return float(vals[lo]) if lo == hi else vals[lo] * (hi - k) + vals[hi] * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='project root with signpost/ and datasets/processed/')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    sys.path.insert(0, str(root))
    from signpost.retrieval import offline_signpost as O
    from signpost.retrieval.signpost_variants import _apply_cue_topb_item
    build = {'chunk': O._chunk_signpost, 'summary': O._summary_signpost,
             'entity': O._entity_signpost, 'relation': O._relation_signpost}
    gp = root / 'datasets/processed' / args.dataset / 'graph.unified.json'
    t0 = time.perf_counter()
    graph = json.loads(gp.read_text(encoding='utf-8'))
    idx = O.GraphIndex(graph)
    objs = [(n['node_type'], n) for n in graph['nodes'] if isinstance(n, dict) and n.get('node_type') in ('chunk', 'summary', 'entity')]
    objs += [('relation', e) for e in graph['edges'] if isinstance(e, dict) and e.get('edge_type') == 'semantic']
    labels = ['inf' if b is None else str(b) for b in BUDGETS]
    ent = {l: defaultdict(list) for l in labels}
    tot_e = dict.fromkeys(labels, 0)
    tot_b = dict.fromkeys(labels, 0)
    for rtype, ref in objs:
        v = build[rtype](idx, ref)
        rt = v.get('result_type', rtype)
        for b, l in zip(BUDGETS, labels):
            if b is None:
                w = v
            else:
                item = {'offline_signpost': copy.deepcopy(v)}
                _apply_cue_topb_item(item, dict(v=b, h=b, s=b, p=b))
                w = item['offline_signpost']
            n = entries(w, rt)
            ent[l][rt].append(n)
            tot_e[l] += n
            tot_b[l] += len(encode(w))
    rep = {'dataset': args.dataset, 'graph_sha256': sha_file(gp), 'script_sha256': sha_file(__file__),
           'offline_builder_sha256': sha_file(O.__file__), 'platform': platform.platform(), 'python': sys.version,
           'objects': len(objs), 'seconds': time.perf_counter() - t0,
           'total_entries': tot_e, 'total_bytes': tot_b,
           'entries_share_of_inf': {l: tot_e[l] / tot_e['inf'] for l in labels},
           'bytes_share_of_inf': {l: tot_b[l] / tot_b['inf'] for l in labels},
           'per_type': {l: {rt: {'n': len(x), 'mean': sum(x) / len(x), 'p95': pct(sorted(x), .95), 'max': max(x)}
                            for rt, x in ent[l].items() if x} for l in labels}}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'x5_{args.dataset}.json').write_text(json.dumps(rep, indent=2))
    print(args.dataset, json.dumps({l: round(rep['entries_share_of_inf'][l], 4) for l in labels}),
          'rel_p95_inf', rep['per_type']['inf'].get('relation', {}).get('p95'), flush=True)


if __name__ == '__main__':
    main()
