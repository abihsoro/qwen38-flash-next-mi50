# tools/gfx906/ — kernel laboratory tooling

Filled by task **S6**: `bench_{dense,moe,qsa,norm,fusion,memory}.py`, seeded with
S4 shapes. Each candidate reports: PyTorch reference, generic ROCm baseline,
gfx906 implementation, numerical error, latency, effective GB/s, occupancy,
register count, emitted ISA. Every change subject to I5 and I6.
