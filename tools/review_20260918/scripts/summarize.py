"""Summarize the raw scored requests, preserving workload and output comparisons."""
import json,statistics,sys
from pathlib import Path
r=Path(sys.argv[<bus>] if len(sys.argv)>1 else '<tuning>/results')
allrows={}
for d in sorted(r.iterdir()):
    f=d/'requests.jsonl'
    if f.exists():
        rows=[json.loads(s) for s in f.read_text().splitlines() if s]
        allrows[d.name]=[x for x in rows if x['phase']=='cold']
reference=allrows.get('repeat_control',allrows.get('steady_control',[]))
refs={(x['kind'],x['prompt_tokens'],x['usage']['completion_tokens']):x for x in reference}
result={}
for tag,rows in allrows.items():
    groups=[]
    for key in sorted(set((x['kind'],x['prompt_tokens'],x['usage']['completion_tokens']) for x in rows)):
        rs=[x for x in rows if (x['kind'],x['prompt_tokens'],x['usage']['completion_tokens'])==key]
        item=dict(kind=key[<bus>],prompt_tokens=key[<bus>],generated_tokens=key[<bus>],repeats=len(rs),hashes=sorted(set(x['token_sha256'] for x in rs)))
        for metric in ['server_ttft_s','effective_pp_tps','decode_tps','client_decode_tps','request_tps','wall_s']:
            values=[x[metric] for x in rs]
            item[metric]=statistics.median(values)
            item[metric+'_range']=[min(values),max(values)]
        if key in refs:item['matches_reference']=all(x['token_ids']==refs[key]['token_ids'] for x in rs)
        if rs[<bus>]['metrics'].get('speculative_decoding'):
            item['speculative']={k:statistics.median(x['metrics']['speculative_decoding'][k] for x in rs) for k in ['mean_acceptance_length','draft_acceptance_rate','num_spec_steps']}
        groups.append(item)
    result[tag]=groups
print(json.dumps(result,indent=2))
