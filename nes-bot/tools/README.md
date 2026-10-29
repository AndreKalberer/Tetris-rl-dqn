# Tools

NES original-cartridge (level 255) work — see [docs/NES-255.md](../docs/NES-255.md):

| Tool | What it does |
|---|---|
| `run255.sh SEED [255\|511\|endless]` | Play one original-game attempt with the current best bot in MesenCE, then replay-verify it. |
| `run-score80.sh SEED [255\|511\|endless]` | The 80M score bot: 8-row tetris stack with 2-piece playouts. Needs 3 planner cores (default mask 13) and 1 emulator core, so one game per 4-core machine. |
| `run-score-vnet.sh SEED [255\|511\|endless]` | The 80M bot plus a learned leaf value (`value-net-v3.npz`) at levels 176-243 while the stack is at most 6 rows. Same cores as `run-score80.sh`. |
| `value_net/gen.py`, `value_net/train.py` | Simulated 1G trajectories for value-net training, and the trainer that exports the `.npz` net (needs PyTorch). |
| `run-score50.sh SEED [255\|511\|endless]` | Same as `run255.sh` with the 50M score bot (CEM tetris weights, 3rd-piece expectation, no spawn wait). |
| `batch255.py` | Run a fixed list of seeds a few at a time with one launcher (status in `batch-status.json`). |
| `zone_sim.py` | Crash-zone simulator: plays levels 150–175 / 246–255 (both laps) with the real decision code; can start from a real board and piece sequence. `--stop-lines 3300` plays a whole lap; `--playout-top` turns on playouts; `--late-prob` makes a share of plans one frame late. |
| `tune_zone.py` | Tune zone weights on the crash-zone simulator. |
| `tune_low_stack_tetris.py` | Tune tetris-building weights used only while the stack is low (simulator). |
| `watch_replay.py` / `watch.cmd [speed] [record]` | Watch a recorded game in a Mesen window with an uncapped score / level / lines panel. |

Simulator tuning of the general evaluator: `python -m nes_original.tune` (tune / compare).

Older tools: `package_*.py` package checkpoints and releases; `verify_nes_reference.py` runs selected original 6502 routines (from the public disassembly) against the simulator.
