#!/usr/bin/env python3
"""X4 features: for every chunk edit batch of e4_incremental.py (same seed, same
sampling sequence, same dirty-set expansion), compute cost-model features that a
maintained store knows without recomputing any view:

  dirty_share       |D| / N
  bytes_share       sum of stored-view bytes over D / over all objects (b = inf)
  entries_share     same with list-entry counts instead of bytes
  degree_share      same with 1 + adjacency degree of the object's endpoints

The e4 dirty-set sizes are asserted against the frozen e4 JSON when given.
No model, ES or network calls.
"""
from __future__ import annotations

import argparse, hashlib, json, platform, random, sys, time
from collections import defaultdict
from pathlib import Path

from signpost.retrieval.offline_signpost import (GraphIndex, _chunk_signpost, _summary_signpost,
                                                 _entity_signpost, _relation_signpost)

CHUNK_FRACTIONS = [0.001, 0.005, 0.01, 0.05, 0.10]
SEED = 20260919


def node_id(node):
    for key in ("node_id", "id", "chunk_id"):
        v = node.get(key)
        if v:
            return str(v)
    return None


def entries(x):
    if isinstance(x, dict):
        return sum(entries(v) for v in x.values())
    if isinstance(x, list):
        return len(x)
    return 0


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--e4', default=None, help='frozen e4 JSON to check dirty-set sizes against')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    graph = json.loads(Path(args.graph).read_text(encoding='utf-8'))
    index = GraphIndex(graph)
    nodes = [n for n in graph.get('nodes', []) if isinstance(n, dict)]
    edges = [e for e in graph.get('edges', []) if isinstance(e, dict)]
    objects = []
    for n in nodes:
        if n.get('node_type') in {'chunk', 'summary', 'entity'}:
            objects.append((n['node_type'], n, node_id(n)))
    for e in edges:
        if e.get('edge_type') == 'semantic':
            objects.append(('relation', e, node_id(e)))
    n_obj = len(objects)
    adj = defaultdict(set)
    for e in edges:
        s, t = e.get('source') or e.get('src') or e.get('from'), e.get('target') or e.get('dst') or e.get('to')
        if s and t:
            adj[str(s)].add(str(t))
            adj[str(t)].add(str(s))
    obj_endpoints = []
    for rtype, ref, oid in objects:
        if rtype == 'relation':
            eps = {str(x) for x in (ref.get('source') or ref.get('src'), ref.get('target') or ref.get('dst')) if x}
        else:
            eps = {oid} if oid else set()
        obj_endpoints.append(eps)
    touched_by = defaultdict(set)
    for i, eps in enumerate(obj_endpoints):
        for v in eps:
            touched_by[v].add(i)
    build = {'chunk': _chunk_signpost, 'summary': _summary_signpost, 'entity': _entity_signpost, 'relation': _relation_signpost}
    w_bytes, w_entries, w_degree = [], [], []
    t0 = time.perf_counter()
    for (rtype, ref, _), eps in zip(objects, obj_endpoints):
        v = build[rtype](index, ref)
        w_bytes.append(len(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()))
        w_entries.append(1 + entries(v))
        w_degree.append(1 + sum(len(adj.get(x, ())) for x in eps))
    t_weights = time.perf_counter() - t0
    tot = {'bytes': sum(w_bytes), 'entries': sum(w_entries), 'degree': sum(w_degree)}

    rng = random.Random(SEED)
    chunk_ids = [node_id(n) for n in nodes if n.get('node_type') == 'chunk' and node_id(n)]
    frozen = None
    if args.e4:
        fr = json.loads(Path(args.e4).read_text())
        frozen = {e['edit_fraction']: e for e in fr['edits'] if e['granularity'] == 'chunk'}
        assert fr['objects'] == n_obj, (fr['objects'], n_obj)
    rows = []
    for frac in CHUNK_FRACTIONS:
        k = max(1, int(round(len(chunk_ids) * frac)))
        delta_v = set(rng.sample(chunk_ids, k))
        dirty_vertices = set(delta_v)
        for v in delta_v:
            dirty_vertices |= adj.get(v, set())
        dirty = set()
        for v in dirty_vertices:
            dirty |= touched_by.get(v, set())
        row = {'edit_fraction': frac, 'units_edited': k, 'dirty_objects': len(dirty),
               'dirty_share': len(dirty) / n_obj,
               'bytes_share': sum(w_bytes[i] for i in dirty) / tot['bytes'],
               'entries_share': sum(w_entries[i] for i in dirty) / tot['entries'],
               'degree_share': sum(w_degree[i] for i in dirty) / tot['degree']}
        if frozen:
            e = frozen[frac]
            assert e['dirty_objects'] == len(dirty), ('dirty-set mismatch', frac, e['dirty_objects'], len(dirty))
            row['measured_share_of_full'] = e['share_of_full']
            row['measured_control_share_of_full'] = e['control_share_of_full']
        rows.append(row)
        print(args.dataset, json.dumps(row), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {'dataset': args.dataset, 'graph': args.graph, 'graph_sha256': sha(args.graph), 'script_sha256': sha(__file__),
           'platform': platform.platform(), 'python': sys.version, 'objects': n_obj, 'chunks': len(chunk_ids),
           'weights_seconds': t_weights, 'totals': tot, 'e4_checked': bool(frozen), 'batches': rows}
    (out / f'x4_{args.dataset}.json').write_text(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
