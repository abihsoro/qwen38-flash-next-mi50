"""Exact-length, salted-cache, token-counted streaming benchmark.

Cold PP is effective prompt tokens / server TTFT, not GPU-only prefill speed.
Decode uses server first-to-last token duration and subtracts the entire first
streamed token burst from the numerator, including under speculative decoding.
Raw stream burst timing and server metrics are retained for independent review.
"""
import argparse, hashlib, json, statistics, time, urllib.request, uuid
from pathlib import Path

def hardware():
    cards=[]
    for p in sorted(Path('/sys/class/drm').glob('card[<bus-range>]*/device')):
        try:
            if (p/'device').read_text().strip()!='0x66a0': continue
            h=next((p/'hwmon').glob('hwmon*'))
            row={'pci':p.resolve().name,'numa':(p/'numa_node').read_text().strip()}
            for key in ('power1_cap','power1_average','power1_input','temp1_input','temp2_input','temp3_input'):
                if (h/key).exists(): row[key]=int((h/key).read_text())
            for key in ('pp_dpm_sclk','pp_dpm_mclk'):
                if (p/key).exists(): row[key]=(p/key).read_text().strip()
            cards.append(row)
        except (OSError,StopIteration): pass
    assert len(cards)==4,cards
    assert all(x['power1_cap']==150000000 for x in cards),cards
    return cards

BASE='http://127.0.0.1:8002'
MODEL='qwen38-flash-next'
TEXTS={
 'prose': ('A hash table stores key and value pairs. Collisions occur when different keys map to the same bucket. '
           'Separate chaining stores multiple entries in a bucket. Open addressing searches other buckets. '
           'Resizing reduces the load factor and requires reinserting existing entries. ',
           '\nExplain these tradeoffs in detail, with practical examples and clear paragraphs.'),
 'code': ('We need a Python function implementing a bounded least recently used cache. It supports get and put, '
          'updates access order, evicts the oldest unused item, and handles replacing existing keys. '
          'Use a dictionary and a doubly linked list. Every operation should take constant expected time. ',
          '\nWrite a complete Python implementation, followed by tests and an explanation of the invariants.'),
 'reasoning': ('A warehouse packs products into boxes. Each box has a weight limit of twenty kilograms. '
               'A small item weighs three kilograms, a medium item five, and a large item eight. '
               'There are twelve small items, nine medium items, and six large items. ',
               '\nFind a packing with as few boxes as possible. Explain the lower bound and verify each box weight.')}

def post(path,payload):
    req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as r:
        raw=r.read()
        return json.loads(raw) if raw else None

def token_ids(text):
    out=post('/tokenize',{'model':MODEL,'prompt':text,'add_special_tokens':False})
    return out.get('tokens') or out['token_ids']

def prompt(kind,n):
    body,end=TEXTS[kind]
    suffix=token_ids(end)
    ids=token_ids(body*max(10,n//20))
    result=ids[:n-len(suffix)]+suffix
    assert len(result)==n and n>len(suffix)
    return result

def one(ids,gen,salt):
    devices_before=hardware()
    body={'model':MODEL,'prompt':ids,'max_tokens':gen,'temperature':0.0,'top_p':1.0,
          'ignore_eos':True,'seed':0,'stream':True,'return_token_ids':True,
          'stream_options':{'include_usage':True},'cache_salt':salt}
    req=urllib.request.Request(BASE+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter(); bursts=[]; tokens=[]; text=''; usage=None; metrics=None
    with urllib.request.urlopen(req,timeout=900) as r:
        for line in r:
            if not line.startswith(b'data:'): continue
            data=line[5:].strip()
            if data==b'[DONE]': break
            chunk=json.loads(data)
            if chunk.get('usage'): usage=chunk['usage']
            if chunk.get('metrics'): metrics=chunk['metrics']
            for c in chunk.get('choices',[]):
                delta=c.get('token_ids') or []
                if delta:
                    bursts.append({'s':time.perf_counter()-start,'n':len(delta)})
                    tokens.extend(delta)
                text+=c.get('text','')
    wall=time.perf_counter()-start
    assert usage and len(tokens)==usage['completion_tokens']==gen,(len(tokens),usage)
    assert usage['prompt_tokens']==len(ids),(len(ids),usage)
    assert metrics and metrics.get('time_to_first_token_ms') is not None,metrics
    first_n=bursts[<bus>]['n']; generation=metrics['generation_time_ms']/1000
    cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
    cold=salt.startswith('cold-') or salt.startswith('warmup-')
    if cold: assert cached==0,usage
    return {'hardware_before':devices_before,'hardware_after':hardware(),
        'wall_s':wall,'client_ttft_s':bursts[<bus>]['s'],
        'server_ttft_s':metrics['time_to_first_token_ms']/1000,
        'effective_pp_tps':len(ids)/(metrics['time_to_first_token_ms']/1000),
        'decode_tps':(len(tokens)-first_n)/generation if generation>0 else None,
        'client_decode_tps':(len(tokens)-first_n)/(bursts[-1]['s']-bursts[<bus>]['s']) if len(bursts)>1 else None,
        'request_tps':len(tokens)/wall,'first_burst_tokens':first_n,
        'usage':usage,'metrics':metrics,'bursts':bursts,'token_ids':tokens,
        'token_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
        'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',required=True)
    p.add_argument('--reps',type=int,default=3); p.add_argument('--gen',type=int,default=128)
    p.add_argument('--sizes',type=int,nargs='+',default=[128,512,1024])
    p.add_argument('--kinds',nargs='+',default=list(TEXTS))
    p.add_argument('--cache-test',action='store_true'); p.add_argument('--profile',action='store_true')
    a=p.parse_args(); root=Path('<tuning>/results')/a.tag
    root.mkdir(parents=True,exist_ok=True)
    out=open(root/'requests.jsonl','a',buffering=1); rows=[]
    for n in a.sizes:
        for kind in a.kinds:
            ids=prompt(kind,n)
            (root/f'prompt_{kind}_{n}.json').write_text(json.dumps(ids))
            # Two full generation warmups exercise the same shapes as scored requests.
            for w in range(2):
                r=one(ids,a.gen,'warmup-'+uuid.uuid4().hex)
                r.update(kind=kind,prompt_tokens=n,phase='warmup',rep=w,tag=a.tag,utc=time.time())
                out.write(json.dumps(r)+'\n')
            if a.profile:
                post('/start_profile',{})
            for rep in range(a.reps):
                r=one(ids,a.gen,'cold-'+uuid.uuid4().hex)
                r.update(kind=kind,prompt_tokens=n,phase='cold',rep=rep,tag=a.tag,utc=time.time())
                rows.append(r); out.write(json.dumps(r)+'\n')
                print(json.dumps({k:r[k] for k in ('tag','kind','prompt_tokens','rep','server_ttft_s','decode_tps','token_sha256')}),flush=True)
            if a.profile:
                post('/stop_profile',{})
            if a.cache_test:
                salt='cached-'+uuid.uuid4().hex
                one(ids,1,salt)
                for rep in range(a.reps):
                    r=one(ids,a.gen,salt)
                    r.update(kind=kind,prompt_tokens=n,phase='cached',rep=rep,tag=a.tag,utc=time.time())
                    rows.append(r); out.write(json.dumps(r)+'\n')
    summary=[]
    for kind,n,phase in sorted(set((r['kind'],r['prompt_tokens'],r['phase']) for r in rows)):
        rs=[r for r in rows if (r['kind'],r['prompt_tokens'],r['phase'])==(kind,n,phase)]
        s={'kind':kind,'prompt_tokens':n,'phase':phase,'n':len(rs),
           'hashes':sorted(set(r['token_sha256'] for r in rs)),
           'cached_tokens':[r['usage'].get('prompt_tokens_details') for r in rs]}
        for metric in ('server_ttft_s','effective_pp_tps','decode_tps','client_decode_tps','request_tps'):
            vs=[r[metric] for r in rs if r[metric] is not None]
            s[metric]=statistics.median(vs) if vs else None
            s[metric+'_range']=[min(vs),max(vs)] if vs else None
        summary.append(s)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETED '+a.tag,flush=True)

if __name__=='__main__': main()
