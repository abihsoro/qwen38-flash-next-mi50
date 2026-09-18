"""Reapply the reviewed 150 W/card contract after a reboot (run on Proxmox)."""
import json
from pathlib import Path
cards=[]
for d in sorted(Path('/sys/class/drm').glob('card[<bus-range>]*/device')):
    if not d.parent.name.removeprefix('card').isdigit():continue
    if (d/'device').read_text().strip()!='0x66a0':continue
    h=next((d/'hwmon').glob('hwmon*'))
    assert int((h/'power1_cap_min').read_text())<=150000000<=int((h/'power1_cap_max').read_text())
    cards.append((d,h))
assert len(cards)==4,'Expected four MI50 devices; no settings changed'
for d,h in cards:
    (h/'power1_cap').write_text('150000000')
    (d/'power_dpm_force_performance_level').write_text('auto')
print(json.dumps([dict(pci=d.resolve().name,power_cap_uw=int((h/'power1_cap').read_text()),dpm=(d/'power_dpm_force_performance_level').read_text().strip()) for d,h in cards],indent=2))
