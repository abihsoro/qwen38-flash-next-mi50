#!/usr/bin/env python3
"""S1 check 06 — torch.compile: exit 0, output matches eager within tolerance."""
import sys

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main() -> int:
    torch.manual_seed(0)
    net = Net().cuda().eval()
    x = torch.randn(64, 256, device="cuda")
    with torch.no_grad():
        eager = net(x)
        compiled = torch.compile(net)(x)
    torch.cuda.synchronize()
    rel = (compiled - eager).abs().max().item() / eager.abs().max().item()
    print(f"compile_vs_eager_rel_err={rel:.3e}")
    ok = rel < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
