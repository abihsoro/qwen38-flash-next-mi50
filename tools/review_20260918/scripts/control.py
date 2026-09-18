import argparse, hashlib, json, os, signal, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path('<tuning>')
PYTHON = '<home>/gfx906-venv/bin/python3'

def stop():
    f = ROOT / 'server.json'
    if not f.exists():
        return
    state = json.loads(f.read_text())
    pid = state['pid']
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        cmd = b''
    if cmd and state.get('process_start_ticks') is not None:
        actual=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[<bus>].split()[<bus>]
        if actual != state['process_start_ticks']:
            raise RuntimeError('PID was reused; refusing to stop an unrelated process')
    if cmd and b'vllm' not in cmd and b'numactl' not in cmd:
        raise RuntimeError('PID belongs to an unexpected process; refusing to stop')
    for sig, delay in ((signal.SIGTERM, 8), (signal.SIGKILL, 2)):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        time.sleep(delay)
    f.unlink()

def start(args):
    if (ROOT / 'server.json').exists():
        raise RuntimeError('Stop the owned server before starting another arm')
    env = os.environ.copy()
    for key in ('LD_LIBRARY_PATH', 'VLLM_TUNED_CONFIG_FOLDER', 'NCCL_P2P_LEVEL', 'VLLM_MI50_TOPK', 'VLLM_MI50_DENSE_INT8', 'PYTORCH_TUNABLEOP_FILENAME', 'PYTORCH_TUNABLEOP_RECORD_UNTUNED'):
        env.pop(key, None)
    env.update(ROCM_PATH='/opt/rocm', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH='<workdir>/vllm-w:<workdir>', ROCR_VISIBLE_DEVICES='0,1,2,3',
        HSA_NO_SCRATCH_RECLAIM='1', VLLM_ROCM_USE_AITER='0',
        TORCH_BLAS_PREFER_HIPBLASLT='0', FLASH_ATTENTION_TRITON_AMD_ENABLE='TRUE',
        PYTORCH_TUNABLEOP_ENABLED='0', PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED='0',
        VLLM_PLE_CPU_OFFLOAD='1',
        VLLM_PLE_QUANT_DIR='<models>/qwen38-flash-next-ple/ples_int4',
        VLLM_PLE_OFFLOAD_READY_TIMEOUT='3600', VLLM_RDNA_DENSE_INT8='1',
        VLLM_RDNA_AR='1', VLLM_DISABLE_COMPILE_CACHE='1', VLLM_ROCM_USE_SKINNY_GEMM='1',
        VLLM_RDNA_AR_BLOCKS=str(args.ar_blocks))
    for kv in args.env:
        k,v = kv.split('=',1); env[k] = v
    cmd = [PYTHON,'-m','vllm.entrypoints.openai.api_server',
        '--model','<models>/models/qwen38-flash-next-awq',
        '--served-model-name','qwen38-flash-next','--dtype','float16',
        '--tensor-parallel-size','4','--enable-expert-parallel','--max-model-len',str(args.max_model_len),
        '--gpu-memory-utilization','0.90','--max-num-seqs','4',
        '--max-num-batched-tokens',str(args.chunk),'-cc.mode=none','-cc.cudagraph_mode=full',
        '--language-model-only','--enable-prefix-caching','--enable-per-request-metrics',
        '--enable-prompt-tokens-details',
        '--host','0.0.0.0','--port','8002']
    if args.mtp:
        cmd += ['--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':args.mtp}),
                '--per-request-spec-decode-metrics','summary']
    if args.profile:
        cmd += ['--profiler-config',json.dumps({'profiler':'torch','torch_profiler_dir':str(ROOT/'results'/args.tag/'trace')})]
    if args.numa:
        cmd = ['numactl','--cpunodebind=1','--preferred=1']+cmd
    out = ROOT/'results'/args.tag
    out.mkdir(parents=True,exist_ok=True)
    log = open(out/'server.log','w')
    p = subprocess.Popen(cmd,cwd='<workdir>',env=env,stdout=log,stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL,start_new_session=True)
    state = {'pid':p.pid,'tag':args.tag,'args':vars(args),'command':cmd,
             'environment':{k:v for k,v in env.items() if k.startswith(('VLLM_','PYTORCH_','ROCR_','NCCL_','TORCH_','HSA_'))},
             'started_at':time.time(),
             'process_start_ticks':Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[<bus>].split()[<bus>],
             'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                 Path('<workdir>/vllm-w/vllm/model_executor/layers/rdna_dense_int8.py'),
                 Path('<workdir>/vllm-w/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py'),
                 Path('<workdir>/vllm-w/vllm/model_executor/layers/fused_moe/router/mi50_topk.py')) if p.exists()}}
    tuning=env.get('VLLM_TUNED_CONFIG_FOLDER')
    if tuning:state['moe_config_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(tuning).glob('*.json')}
    (ROOT/'server.json').write_text(json.dumps(state,indent=2))
    (out/'launch.json').write_text(json.dumps(state,indent=2))
    print(json.dumps({'launched':p.pid,'tag':args.tag}),flush=True)

ap=argparse.ArgumentParser()
ap.add_argument('action',choices=['start','stop','status'])
ap.add_argument('--tag',default='baseline150')
ap.add_argument('--mtp',type=int,default=0)
ap.add_argument('--chunk',type=int,default=1024)
ap.add_argument('--max-model-len',type=int,default=2048)
ap.add_argument('--ar-blocks',type=int,default=0)
ap.add_argument('--numa',action='store_true')
ap.add_argument('--profile',action='store_true')
ap.add_argument('--env',action='append',default=[])
a=ap.parse_args()
if a.action=='start': start(a)
elif a.action=='stop': stop()
else:
    try:
        print(urllib.request.urlopen('http://127.0.0.1:8002/health',timeout=2).status)
    except Exception as e:
        print(type(e).__name__+': '+str(e))
