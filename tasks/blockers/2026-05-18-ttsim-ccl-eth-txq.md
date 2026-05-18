# TT-Sim Blackhole CCL Fabric Blocker

Date: 2026-05-18

## Blocked Work

Active objective: complete T10.1, T10.2, and T10.3.

Current blocking gate:

- T10.1 CCL runtime correctness for all-gather, reduce-scatter, and
  all-to-all.
- Success requires all three collectives to pass the repository
  `BlackholeModule + TT-Sim bf16 + host reference` numerical comparison gate.
- T10.2 and T10.3 are ordered after the T10.1d runtime value gate and must not
  be counted complete before it passes.

## Reproduction

From `/root/dev/vibe_dsl`:

```bash
scripts/probe_tt_sim_ccl.sh
```

The probe sources the repository TT-Sim setup, sets the Blackhole P300
mock-cluster descriptor, confirms that a no-fabric `1x2` mesh opens, and then
runs the minimal bf16 CCL value gate for:

- `ttnn.all_gather`
- `ttnn.reduce_scatter`
- `ttnn.experimental.all_to_all_async`

Exit code `0` means all three collective outputs match the host references.

## Current Result

The no-fabric mesh phase succeeds:

```text
mesh_without_fabric_ok=true
mesh_num_devices=2
```

The fabric CCL phase reaches fabric initialization on both simulated devices,
then fails before completing the first all-gather step:

```text
UnimplementedFunctionality: eth_txq_regs_wr32: eth_txq_cmd=0x2
probe_status=fabric_ccl_unsupported
unsupported_reason=eth_txq_cmd=0x2
```

`0x2` corresponds to `ETH_TXQ_CMD_START_DATA`.

## Ruled Out

- Running without fabric config is not a fallback; the all-gather path fails
  with `Trying to get un-initialized fabric context`.
- `TTSIM_SEMIHOSTING=1` does not change the failure.
- Replacing the local simulator with upstream TT-Sim `v1.6.1` through
  `TT_SIM_LIB_OVERRIDE=/tmp/ttsim-v1.6.1/libttsim.so` reaches the same
  `eth_txq_cmd=0x2` fatal.
- The public `tenstorrent/ttsim` repository currently publishes release
  binaries and documentation only; the source is not available for local
  patching.
- Public GitHub searches for `repo:tenstorrent/ttsim eth_txq`,
  `repo:tenstorrent/ttsim CCL Blackhole`, and
  `repo:tenstorrent/ttsim fabric` returned no existing workaround.
- This host has no real Tenstorrent device path available: no `tt-smi`,
  no Tenstorrent PCI device in `lspci`, and no `/dev/tenstorrent*` device.

## Required External Input

One of the following is required before T10.1 can continue:

- A Blackhole TT-Sim binary that supports the fabric TXQ data command path hit
  by `ETH_TXQ_CMD_START_DATA`; or
- Access to a real multi-device Blackhole target that can run the same bf16
  CCL value gate.

After either input is available, rerun:

```bash
scripts/probe_tt_sim_ccl.sh
```

Then continue with T10.1c/T10.1d only if the probe reaches CCL value execution
and reports numerical correctness for all-gather, reduce-scatter, and
all-to-all.
