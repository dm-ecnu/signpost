#!/usr/bin/env python3
"""Same-function view materialization benchmark. No model/API/ES calls.
Reconstruct exact offline-view request sequences from frozen search subqueries,
using the pilot's unchanged local top-5 retrievers. Replay ONLY view service.
Graph loading/indexing are common and reported separately. Cache capacity and
storage count serialized JSON value bytes, not Python overhead or total RSS.
Cold means empty application cache; OS cache is neither cleared nor claimed cold.
"""
import argparse, collections, gc, hashlib, json, os, platform, random, statistics, sys, time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def encode(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--dataset',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--repeats',type=int,default=5);ap.add_argument('--cycles',type=int,default=3)
    args=ap.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    from signpost.retrieval import offline_signpost as O
    from signpost.retrieval.signpost_variants import _apply_cue_topb_item
    from signpost.agent.tools import _local_chunk_search, _local_graph_search
    gp=root/'datasets/processed'/args.dataset/'graph.unified.json'
    cp=gp.parent/'chunks.jsonl';pp=root/'outputs/e1e2'/('full__'+args.dataset+'.jsonl')
    t=time.perf_counter();graph=json.loads(gp.read_text());load_s=time.perf_counter()-t
    t=time.perf_counter();idx=O.GraphIndex(graph);index_s=time.perf_counter()-t
    builders={'chunk':O._chunk_signpost,'summary':O._summary_signpost,'entity':O._entity_signpost,'relation':O._relation_signpost}
    objects={};aliases={}
    for n in graph['nodes']:
        if n.get('node_type') in builders:
            key=n['node_id'];objects[key]=(n['node_type'],n);aliases[key]=key
            if n.get('chunk_id'):aliases[n['chunk_id']]=key
    for e in graph['edges']:
        if e.get('edge_type')=='semantic':
            key=O._edge_id(e);objects[key]=('relation',e);aliases[key]=key
            aliases[str(e['source'])+'->'+str(e['target'])]=key
    def resolve(item):
        for k in ['node_id','chunk_id','edge_id','id']:
            if item.get(k) in aliases:return aliases[item[k]]
        if item.get('source') and item.get('target'):
            return aliases.get(str(item['source'])+'->'+str(item['target']))
        raise ValueError('Unresolved search result '+str(item)[:200])
    manifest={'dataset':args.dataset,'graph_sha256':sha(gp),'predictions_sha256':sha(pp),'chunks_sha256':sha(cp),
              'script_sha256':sha(__file__),'offline_builder_sha256':sha(O.__file__),
              'platform':platform.platform(),'python':sys.version,'cpu':platform.processor(),
              'load_seconds':load_s,'index_seconds':index_s,'objects':len(objects),'repeats':args.repeats,'cycles':args.cycles,
              'boundary':'view construction, prefix selection, canonical JSON serialization, cache lookup; excludes common retrieval, graph load/index, network, LLM, source reads',
              'cache_units':'serialized value bytes; excludes key/container/Python overhead',
              'cold_definition':'empty application object cache; OS cache uncontrolled',
              'trace_source':'same local top-5 chunk/summary/graph retrieval applied to frozen full-arm knowledge_search inputs'}
    tracepath=out/(args.dataset+'.trace.json')
    if tracepath.exists():
        old=json.loads(tracepath.read_text());assert old['graph_sha256']==manifest['graph_sha256'] and old['predictions_sha256']==manifest['predictions_sha256']
        trace=old['trace']
    else:
        trace=[]; t=time.perf_counter()
        for i,line in enumerate(pp.open()):
            r=json.loads(line);ids=[];queries=[]
            for event in r['trace']:
                if event.get('event_type')=='tool_call' and event.get('tool')=='knowledge_search':
                    q=event['input']['query'];queries.append(q)
                    items=_local_chunk_search(cp,q,5)+_local_graph_search(graph,q,['summary'],5)+_local_graph_search(graph,q,['entity','relation'],5)
                    ids.extend(resolve(item) for item in items)
            assert ids, r['question_id']
            trace.append({'question_id':r['question_id'],'objects':ids,'search_queries':queries})
            if (i+1)%10==0: print(args.dataset,'trace',i+1,'seconds',round(time.perf_counter()-t,1),flush=True)
        tracepath.write_text(json.dumps({'graph_sha256':manifest['graph_sha256'],'predictions_sha256':manifest['predictions_sha256'],'trace':trace},ensure_ascii=False))
    manifest['trace_sha256']=sha(tracepath);manifest['queries']=len(trace);manifest['requests']=sum(len(r['objects']) for r in trace)
    manifest['unique_requested_objects']=len({k for r in trace for k in r['objects']})
    (out/(args.dataset+'.manifest.json')).write_text(json.dumps(manifest,indent=2))
    print(args.dataset,'TRACE READY',manifest['queries'],manifest['requests'],manifest['unique_requested_objects'],flush=True)
    rows=[]
    def make(k,b):
        kind,ref=objects[k];v=builders[kind](idx,ref)
        if b:_apply_cue_topb_item({'offline_signpost':v},dict(v=b,h=b,s=b,p=b))
        return encode(v)
    modes=['eager','lazy_unbounded','lazy_lru_1pct','lazy_lru_10pct','no_cache']
    resultpath=out/(args.dataset+'.runs.jsonl')
    assert not resultpath.exists(), 'Refusing to overwrite completed/partial benchmark; use a new out directory'
    for b in [0,8]:
        # Reference construction verifies the exact common function and measures storage.
        t=time.perf_counter();reference={k:make(k,b) for k in objects};ref_build=time.perf_counter()-t
        size=sum(map(len,reference.values()))
        expected=[hashlib.sha256(b''.join(reference[k] for k in q['objects'])).hexdigest() for q in trace]
        # Check public production dispatcher semantics on all workload objects (outside timing).
        if hasattr(O, "_INDEX_CACHE"): O._INDEX_CACHE[id(graph)]=(graph,idx)
        for k in {k for q in trace for k in q['objects']}:
            kind,ref=objects[k];v=O.build_offline_signpost(graph,ref)
            if b:_apply_cue_topb_item({'offline_signpost':v},dict(v=b,h=b,s=b,p=b))
            assert encode(v)==reference[k],('dispatcher mismatch',k)
        del reference;gc.collect()
        print(args.dataset,'budget',b,'payload MB',round(size/1e6,2),'reference build',round(ref_build,3),flush=True)
        for rep in range(args.repeats):
            order=modes.copy();random.Random(20260920+rep).shuffle(order)
            for mode in order:
                gc.collect();cache=collections.OrderedDict();used=0;peak=0;build_s=0
                limit= size//100 if mode=='lazy_lru_1pct' else size//10 if mode=='lazy_lru_10pct' else None
                if mode=='eager':
                    t=time.perf_counter();cache={k:make(k,b) for k in objects};build_s=time.perf_counter()-t
                    used=peak=size
                for cycle in range(args.cycles):
                    times=[];hits=misses=0
                    for qi,q in enumerate(trace):
                        values=[];t=time.perf_counter()
                        for k in q['objects']:
                            if k in cache:
                                v=cache[k];hits+=1
                                if mode.startswith('lazy_lru'):cache.move_to_end(k)
                            else:
                                v=make(k,b);misses+=1
                                if mode!='no_cache' and (limit is None or len(v)<=limit):
                                    if limit is not None:
                                        while used+len(v)>limit:
                                            _,ev=cache.popitem(last=False);used-=len(ev)
                                    cache[k]=v;used+=len(v);peak=max(peak,used)
                            values.append(v)
                        times.append(time.perf_counter()-t)
                        assert hashlib.sha256(b''.join(values)).hexdigest()==expected[qi],('replay mismatch',mode,b,rep,cycle,qi)
                    st=sorted(times)
                    row={'dataset':args.dataset,'budget':b,'repeat':rep,'mode':mode,'cycle':cycle,'queries':len(trace),
                         'upfront_seconds':build_s if cycle==0 else 0,'service_seconds':sum(times),
                         'query_p50_ms':statistics.median(times)*1000,'query_p95_ms':st[int(.95*(len(st)-1))]*1000,
                         'cache_hits':hits,'cache_misses':misses,'payload_resident_bytes':used,'payload_peak_bytes':peak,
                         'full_view_payload_bytes':size,'equal_to_eager':True}
                    rows.append(row)
                    with resultpath.open('a') as f:f.write(json.dumps(row)+'\n')
                print(args.dataset,'b',b,'repeat',rep,mode,'build',round(build_s,4),'last service',round(sum(times),4),flush=True)
                del cache;gc.collect()
    (out/(args.dataset+'.complete.json')).write_text(json.dumps({'rows':len(rows),'runs_sha256':sha(resultpath),'all_equal':True}))

if __name__=='__main__':main()
