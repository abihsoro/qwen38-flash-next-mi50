# Host-side units (operator installs)

clear-pex-acs.service - clears the PEX88096's ACS control bits on the four
downstream ports at boot (D073/D074). The kernel's pcie_acs_override only
affects IOMMU grouping; the SILICON must be cleared for P2P routing, and any
PCI reset/reboot re-asserts it. Install:
  sudo cp tools/host/clear-pex-acs.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable clear-pex-acs.service
Verify after boot: lspci -vvv -s <port> | grep ACSCtl  (expect ReqRedir-).
G3 at M0 remains the bandwidth determination.
