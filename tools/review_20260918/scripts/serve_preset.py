"""Start a reviewed preset, replacing only a server owned by this campaign."""
import argparse,json,socket,subprocess,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from bench import hardware
r=Path('<tuning>');py=sys.executable
ap=argparse.ArgumentParser();ap.add_argument('preset',nargs='?',default='balanced');ap.add_argument('--stop',action='store_true');a=ap.parse_args()
if a.stop:
    raise SystemExit(subprocess.call([py,str(r/'control.py'),'stop']))
presets=json.loads((r/'presets.json').read_text())
if a.preset not in presets:raise SystemExit('Choose one of: '+', '.join(presets))
hardware()  # Fail before changing the running server if its 150 W contract is absent.
subprocess.run([py,str(r/'control.py'),'stop'],check=True)
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',8002))==0:
        raise SystemExit('Port 8002 belongs to another server; no unrelated process was stopped.')
tag='service_'+a.preset+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
subprocess.run([py,str(r/'control.py'),'start','--tag',tag,*presets[a.preset]['arguments']],check=True)
print(presets[a.preset]['description'],flush=True)
for i in range(600):
    state=json.loads((r/'server.json').read_text())
    status=Path('/proc')/str(state['pid'])/'status'
    if not status.exists() or '\nState:\tZ' in status.read_text():
        raise SystemExit('Server exited; inspect '+str(r/'results'/tag/'server.log'))
    try:
        ready=urllib.request.urlopen('http://127.0.0.1:8002/health',timeout=2).status==200
        prefault='PLE sidecar prefaulted into the page cache:' in (r/'results'/tag/'server.log').read_text()
        if ready and prefault:
            print('Ready on port 8002; launch and logs: '+str(r/'results'/tag),flush=True)
            break
    except (OSError,ValueError):pass
    if i%30==0:print('Loading model and warming sidecar...',flush=True)
    time.sleep(2)
else:raise SystemExit('Startup timed out; inspect '+str(r/'results'/tag/'server.log'))
