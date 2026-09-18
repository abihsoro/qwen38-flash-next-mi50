import json,subprocess, sys, time, urllib.request
from pathlib import Path

root=Path('<tuning>')
for i in range(600):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8002/health',timeout=2) as r:
            if r.status==200:
                state=json.loads((root/'server.json').read_text())
                log=(root/'results'/state['tag']/'server.log').read_text()
                if 'PLE sidecar prefaulted into the page cache:' in log:break
    except Exception: pass
    statefile=root/'server.json'
    if statefile.exists():
        state=json.loads(statefile.read_text())
        status=Path('/proc')/str(state['pid'])/'status'
        if not status.exists() or '\nState:\tZ' in status.read_text():
            raise RuntimeError('Server exited; inspect '+str(root/'results'/state['tag']/'server.log'))
    if i%30==0: print('Waiting for model readiness...',flush=True)
    time.sleep(2)
else: raise RuntimeError('Model did not become ready in 20 minutes')
print('Model ready; benchmark starting',flush=True)
raise SystemExit(subprocess.call([sys.executable,'-u',str(root/'bench.py')]+sys.argv[1:]))
