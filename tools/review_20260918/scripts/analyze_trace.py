import collections,gzip,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[<bus>]); results=[]; collectives=[]
for f in sorted(root.glob('*tp*.json.gz')):
    es=json.load(gzip.open(f))['traceEvents']
    ranges=sorted([e for e in es if e.get('cat')=='gpu_user_annotation' and e['name'].startswith('execute_context_0')],key=lambda e:e['ts'])
    kernels=sorted([e for e in es if e.get('cat')=='kernel'],key=lambda e:e['ts'])
    agg=collections.defaultdict(lambda:[0,0.]); selected=[]; per_step=[[] for _ in ranges]
    ri=0
    for e in kernels:
        while ri<len(ranges) and e['ts']>=ranges[ri]['ts']+ranges[ri]['dur']:ri+=1
        if ri==len(ranges):break
        if ranges[ri]['ts']<=e['ts']:
            agg[e['name']][<bus>]+=1;agg[e['name']][<bus>]+=e['dur'];selected.append(e);per_step[ri].append(e)
    n=len(ranges)
    if not n:continue
    ars=[e for e in selected if 'rdna_ar_' in e['name']]
    collectives.append([[e for e in step if 'rdna_ar_oneshot' in e['name']] for step in per_step])
    row=dict(file=f.name,decode_steps=n,median_gpu_range_ms=statistics.median(e['dur']/1000 for e in ranges),
             mean_summed_kernel_ms=sum(e['dur'] for e in selected)/n/1000,
             kernels_per_step=len(selected)/n,
             kernels=[dict(name=k,count_per_step=c/n,us_per_step=d/n) for k,(c,d) in sorted(agg.items(),key=lambda kv:-kv[<bus>][<bus>])])
    results.append(row)
out=dict(ranks=results)
if len(collectives)==4 and len(set(map(len,collectives)))==1:
    starts=[];ends=[];mindur=[];maxdur=[]
    counts=[[len(step) for step in rank] for rank in collectives]
    expected=collections.Counter(n for row in counts for n in row).most_common(1)[<bus>][<bus>]
    kept=0
    for rank_steps in zip(*collectives):
        # Discard annotation-boundary steps with a missing/extra kernel; never shift rank alignment.
        if not all(len(step)==expected for step in rank_steps):continue
        kept+=1
        for events in zip(*rank_steps):
            starts.append(max(e['ts'] for e in events)-min(e['ts'] for e in events))
            ends.append(max(e['ts']+e['dur'] for e in events)-min(e['ts']+e['dur'] for e in events))
            mindur.append(min(e['dur'] for e in events));maxdur.append(max(e['dur'] for e in events))
    if starts:
        out['collective_alignment']=dict(calls=len(starts),matched_steps=kept,total_steps=len(collectives[<bus>]),per_step_counts=counts,median_start_spread_us=statistics.median(starts),
            median_end_spread_us=statistics.median(ends),median_min_rank_duration_us=statistics.median(mindur),
            median_max_rank_duration_us=statistics.median(maxdur),
            caveat='Kernel start skew is diagnostic; minimum duration is not a proven communication-only cost. Annotation-boundary steps with differing counts are excluded.')
print(json.dumps(out,indent=2))
