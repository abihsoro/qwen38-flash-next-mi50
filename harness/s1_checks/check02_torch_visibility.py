#!/usr/bin/env python3
"""S1 check 02 — torch device visibility: device_count()==1, arch string matches.

Facts (arch list, device name, capability) are printed and recorded in the row
notes. PASS requires device_count == 1 and a gfx906/Vega indication.
"""
import sys

import torch


def main() -> int:
    count = torch.cuda.device_count()
    facts = {"device_count": count}
    if count >= 1:
        facts["device_name"] = torch.cuda.get_device_name(0)
        try:
            facts["capability"] = tuple(torch.cuda.get_device_capability(0))
        except Exception as exc:
            facts["capability_error"] = str(exc)
        facts["arch_list"] = torch.cuda.get_arch_list()
    print(facts)
    name = str(facts.get("device_name", ""))
    arch_list = facts.get("arch_list", [])
    ok = count == 1 and ("gfx906" in arch_list or "Vega" in name or "gfx906" in name)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
