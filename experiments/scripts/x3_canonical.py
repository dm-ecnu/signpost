#!/usr/bin/env python3
"""X3: graph-serialization invariance of navigation views, with and without
canonical load order. No model/API/ES calls.

Variants per corpus (b=inf and b=8 derived from the same built view):
  O   original array order          S_k  arrays shuffled with seed k
  CO  canonical(original)           CS_k canonical(shuffled seed k)
Reports byte identity and order-free membership identity for O vs S_k
(reproduces the paper's shuffle test), CO vs CS_k (canonical load order;
expected 100%), and O vs CO (how much canonicalization changes the views
the rest of the paper measured).
"""
import argparse, copy, hashlib, json, platform, random, sys, time
from pathlib import Path


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def encode(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def order_free(x):
    if isinstance(x, dict):
        return {k: order_free(v) for k, v in x.items()}
    if isinstance(x, list):
        return sorted((order_free(v) for v in x), key=lambda v: encode(v))
    return x


def canonical(graph):
    g = dict(graph)
    g['nodes'] = sorted(graph.get('nodes', []), key=lambda n: (str(n.get('node_id')), encode(n)))
    g['edges'] = sorted(graph.get('edges', []), key=lambda e: (str(e.get('edge_type')), str(e.get('source')), str(e.get('target')), encode(e)))
    return g


def shuffled(graph, seed):
    g = dict(graph)
    rng = random.Random(seed)
    g['nodes'] = list(graph.get('nodes', []))
    g['edges'] = list(graph.get('edges', []))
    rng.shuffle(g['nodes'])
    rng.shuffle(g['edges'])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    sys.path.insert(0, str(root))
    from signpost.retrieval import offline_signpost as O
    from signpost.retrieval.signpost_variants import _apply_cue_topb_item
    builders = {'chunk': O._chunk_signpost, 'summary': O._summary_signpost,
                'entity': O._entity_signpost, 'relation': O._relation_signpost}

    def views(graph):
        """Return {object key: (sha_inf, sha_b8, free_inf, free_b8)} over served objects."""
        idx = O.GraphIndex(graph)
        out = {}
        for n in graph['nodes']:
            if n.get('node_type') in builders:
                out[n['node_id']] = ('node', n)
        for e in graph['edges']:
            if e.get('edge_type') == 'semantic':
                out[O._edge_id(e)] = ('relation', e)
        res = {}
        for k, (kind, ref) in out.items():
            v = builders[n_kind(kind, ref)](idx, ref)
            vb = copy.deepcopy(v)
            _apply_cue_topb_item({'offline_signpost': vb}, dict(v=8, h=8, s=8, p=8))
            res[k] = (hashlib.sha256(encode(v)).digest(), hashlib.sha256(encode(vb)).digest(),
                      hashlib.sha256(encode(order_free(v))).digest(), hashlib.sha256(encode(order_free(vb))).digest())
        return res

    def n_kind(kind, ref):
        return 'relation' if kind == 'relation' else ref['node_type']

    def compare(a, b):
        keys = set(a) | set(b)
        common = set(a) & set(b)
        r = {'objects_a': len(a), 'objects_b': len(b), 'common': len(common), 'key_mismatch': len(keys - common)}
        for i, name in enumerate(['bytes_inf', 'bytes_b8', 'members_inf', 'members_b8']):
            same = sum(1 for k in common if a[k][i] == b[k][i])
            r[name + '_identical'] = same
            r[name + '_identical_pct'] = 100.0 * same / len(keys) if keys else 100.0
        return r

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {'script_sha256': sha_file(__file__), 'offline_builder_sha256': sha_file(O.__file__),
              'platform': platform.platform(), 'python': sys.version, 'seeds': args.seeds,
              'canonical_order': 'nodes by (node_id, canonical JSON); edges by (edge_type, source, target, canonical JSON)',
              'membership': 'every list sorted by canonical JSON of its elements before hashing',
              'datasets': {}}
    for ds in args.datasets:
        gp = root / 'datasets/processed' / ds / 'graph.unified.json'
        t = time.perf_counter()
        graph = json.loads(gp.read_text())
        d = {'graph_sha256': sha_file(gp), 'nodes': len(graph['nodes']), 'edges': len(graph['edges']), 'load_s': time.perf_counter() - t}
        t = time.perf_counter(); vo = views(graph); d['build_s_original'] = time.perf_counter() - t
        vco = views(canonical(graph))
        d['original_vs_canonical'] = compare(vo, vco)
        d['shuffle'] = {}
        for s in args.seeds:
            g = shuffled(graph, s)
            vs = views(g)
            vcs = views(canonical(g))
            d['shuffle'][str(s)] = {'original_vs_shuffled': compare(vo, vs), 'canonical_vs_canonical_shuffled': compare(vco, vcs)}
            del g, vs, vcs
            print(ds, 'seed', s, json.dumps(d['shuffle'][str(s)]['original_vs_shuffled']['bytes_inf_identical_pct']),
                  json.dumps(d['shuffle'][str(s)]['canonical_vs_canonical_shuffled']['bytes_inf_identical_pct']), flush=True)
        report['datasets'][ds] = d
        (out / 'x3_canonical.json').write_text(json.dumps(report, indent=2))
        print(ds, 'done', round(time.perf_counter() - t, 1), 's', flush=True)
        del graph, vo, vco
    report['complete'] = True
    (out / 'x3_canonical.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
