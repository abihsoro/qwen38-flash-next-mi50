# M0 decision tree (pre-written — decide now, execute at M0)

Prepared 2026-09-02 from the single-GPU phase + the fork's documented failure
modes. Every branch names the DETECTION (how you know), the DIAGNOSIS (what it
means), and the ACTION (what to do). Deciding these now means day-one with
four cards is execution, not judgment under time pressure.

## 0. Pre-flight (before any P2P test)

- [ ] Host: clear-pex-acs.service installed AND survives a reboot + rescan
      (journal shows "post-clear ... ReqRedir-" for all four ports). If it
      reverts silently, STOP-A is not done — M0's P2P numbers would be
      root-complex traffic and G3 would false-pass.
- [ ] Kernel cmdline staged with the T6 line + pcie_acs_override=downstream,
      multifunction (N-A1); confirm amdgpu params took (/sys/module/amdgpu/
      parameters/) and note pcie_gen_cap may be a no-op in the guest.
- [ ] All four cards enumerate behind the switch (lspci: four 18xx/1xxx BDFs,
      one per downstream port 0c:00/04/08/0c). Confirm each is in its OWN
      IOMMU group (28-31) and bound to vfio-pci.
- [ ] ./run_m0.sh (the suite, NOT the skips — all four GPUs present).

## 1. Peer write works but bandwidth says root-complex redirect

**Detection:** t01 P2P bandwidth pairs are far below the switch-internal
expectation (e.g., store << 14 GB/s, or symmetric-low where the fork saw
store >> load), even though peer ACCESS succeeds (hipDeviceCanAccessPeer=1)
and the IOMMU groups look right.

**Diagnosis:** the ACS silicon clear did not stick (a rescan/reboot re-asserted
it after the unit ran, or the unit failed silently before the journal read).
pcie_acs_override only fixed the GROUPING; the routing is still redirecting
upstream.

**Action:**
1. Re-read ACSCtl on all four ports (lspci -vvv | grep ACSCtl). If ReqRedir+
   is back → the unit didn't survive the reboot: fix the unit ordering
   (After=pcie enumeration is complete; Before=VFIO binds).
2. setpci the clear manually, re-run t01 immediately (before any rescan) to
   confirm the bandwidth jumps. If it does → the unit timing is the bug.
3. If the clear STICKS but bandwidth is still low → the switch's
   downstream-to-downstream routing is restricted by something else (the
   PEX's port config, a retimer) → escalate to the switch's configuration
   (port-to-port routing enable).

## 2. One card negotiates a different link width

**Detection:** t01 or lspci shows one card at x8/x4 while the other three are
x16 (or a different speed grade).

**Diagnosis:** likely physical (seating, the slot's connection, a marginal
riser/port on the switch) rather than config — the other three trained
identically.

**Action:**
1. Reseat the odd card, re-check. (The MI50 is Gen3 x16 — ALL cards should
   show 8GT/s x16; a Gen4-negotiating card would be a red flag on its own.)
2. If it persists at a lower width: record it, and measure the actual impact
   — the P2P pairs involving that card run slower, but the decode step's
   traffic is small; decide whether it blocks M1 by whether the collective
   timing (G5-relevant) degrades beyond the budget. A narrow card may be
   acceptable for bring-up and fixed at the next maintenance window.
3. Do NOT hold M0/M1 on it unless the collective budget breaks.

## 3. Large BAR fails on cards 2-4 but not card 1

**Detection:** one card's 32 GB BAR0 maps; the others report a truncated BAR
or the guest fails to see full VRAM; dmesg shows the BAR being clipped.

**Diagnosis:** the BIOS's bus/resource allocation to the switch's downstream
ports is uneven (the x8x8/bus-window quirk from D077 can affect per-port
resource windows), or Above 4G/ReBAR coverage is partial for the later ports.

**Action:**
1. Read each card's BAR0 size (lspci -vv) — compare against card 1's 32 GB.
2. If truncated: the BIOS resource window for those ports is the cause —
   check the slot's bus allocation (secondary/subordinate ranges per
   downstream port). The ports with smaller subordinate ranges are the
   suspects; confirm all four got the full range (D070's fix pattern).
3. Re-check after any BIOS setting change. A 32 GB BAR is REQUIRED for the
   card to work (the model's weights + KV per card at TP4 fit only with the
   full aperture) — this one is blocking, unlike #2.

## 4. G5 budget miss (collectives > 34 µs at the dominant size)

**Detection:** t03/t05 (RCCL sweep + comparator) or the M2 timing shows the
all-reduce above budget.

**Diagnosis (D066/D079):** fixed cost was ~15-20 µs separate-launch; the fused
persistent kernel validated at 1.85 µs/step single-GPU. If the four-card
number still misses, the gap is the peer-wait/transport semantics, not the
reduce.

**Action:**
1. If the fused shape (poll+reduce+announce in one resident kernel) is not yet
   in the M2 implementation → adopt it first (D079's validated shape).
2. If it is and still misses: G5 is a t/s-adjustment gate, NOT a STOP (D083
   wording). Record the number, adjust the t/s target, continue — do not
   treat it as a project stop.

## 5. Anything else that "looks wrong" at M0

**Rule:** every unexpected number gets (a) a results row with the exact
conditions, (b) a decision entry, (c) a re-run with a NEGATIVE control where
possible (per the D3 discipline). No number is explained by assertion; no
anomaly is waved off because "it's probably the switch."
