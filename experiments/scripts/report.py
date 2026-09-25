import json,statistics,hashlib
from pathlib import Path
root=Path('signpost/results/efficiency-20260920');out=[];man=[]
for done in sorted(root.glob('*.complete.json')):
 ds=done.name.removesuffix('.complete.json');rp=root/(ds+'.runs.jsonl');mp=root/(ds+'.manifest.json')
 marker=json.loads(done.read_text());assert hashlib.sha256(rp.read_bytes()).hexdigest()==marker['runs_sha256']
 rows=[json.loads(l) for l in rp.read_text().splitlines()];m=json.loads(mp.read_text());man.append(m)
 assert len(rows)==marker['rows'] and all(r['equal_to_eager'] for r in rows)
 for b in [0,8]:
  for mode in ['eager','lazy_unbounded','lazy_lru_10pct','lazy_lru_1pct','no_cache']:
   r=[x for x in rows if x['budget']==b and x['mode']==mode]
   first=[x for x in r if x['cycle']==0];warm=[x for x in r if x['cycle']==m['cycles']-1]
   assert len(first)==len(warm)==m['repeats']
   median=lambda data,fn:statistics.median(fn(x) for x in data)
   total=[sum(x['upfront_seconds']+x['service_seconds'] for x in r if x['repeat']==rep) for rep in range(m['repeats'])]
   out.append({'dataset':ds,'budget':b,'mode':mode,'queries':m['queries'],'requests':m['requests'],
     'objects':m['objects'],'unique_objects':m['unique_requested_objects'],
     'cold_total_s':median(first,lambda x:x['upfront_seconds']+x['service_seconds']),
     'cold_service_s':median(first,lambda x:x['service_seconds']),
     'three_cycles_total_s':statistics.median(total),
     'warm_ms_per_query':median(warm,lambda x:x['service_seconds']/x['queries']*1000),
     'warm_p95_ms':median(warm,lambda x:x['query_p95_ms']),
     'cold_payload_MB':median(first,lambda x:x['payload_resident_bytes']/1e6),
     'cold_cache_misses':median(first,lambda x:x['cache_misses'])})
(root/'summary.json').write_text(json.dumps({'completed_datasets':len(man),'results':out},indent=2))
lines=['# 同函数导航计算效率实测','','每套语料用冻结的 full 回答轨迹中的搜索子问题，重放原检索器以恢复导航请求。所有策略共用图与函数、相同条目上限，逐查询检验输出字节完全一致。','每配置 5 次重复、每次 3 遍相同查询，方法顺序随机。表中为中位数。冷缓存指应用缓存初始为空；未清 OS 缓存。','首次总成本包含全量提前生成；暖态是第三遍。只计导航构建、JSON 序列化与缓存访问，不计共用图载入/索引、检索、模型和读源成本。MB 只计序列化导航值，不是实际 RSS。','','|数据|上限|策略|首次总秒|三遍总秒|暖态毫秒/问|导航 MB|','|---|---|---|---:|---:|---:|---:|']
for r in out:
 lines.append(f"|{r['dataset']}|{r['budget'] or '完整'}|{r['mode']}|{r['cold_total_s']:.4f}|{r['three_cycles_total_s']:.4f}|{r['warm_ms_per_query']:.4f}|{r['cold_payload_MB']:.3f}|")
lines+=['','## 已完成语料的覆盖','']
for m in man:lines.append(f"- {m['dataset']}: {m['queries']} 问，{m['requests']} 次请求，访问 {m['unique_requested_objects']}/{m['objects']} 个对象；图读取 {m['load_seconds']:.3f}s、索引 {m['index_seconds']:.3f}s，二者作为共用开销单列。")
lines+=['','## 解释边界','','这组实验能检验同函数全量预建是否优于对象缓存，不能证明新控制器质量、端到端提速或更新后的缓存正确性。固定查询的重复回放只代表测得的工作负载；不能外推任意新查询与更新频率。现有基底图保持不变，未在本轮比较磁盘持久化和数据库实现。']
Path('signpost/notes/EFFICIENCY-RESULTS-2026-09-20.md').write_text('\n'.join(lines)+'\n')
print('Completed datasets:',len(man))
for m in man:
 r=[x for x in out if x['dataset']==m['dataset'] and x['budget']==0];d={x['mode']:x for x in r};e=d['eager'];l=d['lazy_unbounded']
 print(m['dataset'],'eager/lazy cold',round(e['cold_total_s']/l['cold_total_s'],2),'lazy payload %',round(100*l['cold_payload_MB']/e['cold_payload_MB'],2))
