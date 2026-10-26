# Tools

NES original-cartridge (level 255) work — see [docs/NES-255.md](../docs/NES-255.md):

| Tool | What it does |
|---|---|
| `run255.sh SEED [255\|511\|endless]` | Play one original-game attempt with the current best bot in MesenCE, then replay-verify it. |
| `run-score50.sh SEED [255\|511\|endless]` | Same as `run255.sh` with the 50M score bot (CEM tetris weights, 3rd-piece expectation, no spawn wait). |
| `batch255.py` | Run a fixed list of seeds a few at a time with one launcher (status in `batch-status.json`). |
| `zone_sim.py` | Crash-zone simulator: plays levels 150–175 / 246–255 (both laps) with the real decision code; can start from a real board and piece sequence. |
| `tune_zone.py` | Tune zone weights on the crash-zone simulator. |
| `tune_low_stack_tetris.py` | Tune tetris-building weights used only while the stack is low (simulator). |
| `watch_replay.py` / `watch.cmd [speed] [record]` | Watch a recorded game in a Mesen window with an uncapped score / level / lines panel. |

Simulator tuning of the general evaluator: `python -m nes_original.tune` (tune / compare).

Older tools: `package_*.py` package checkpoints and releases; `verify_nes_reference.py` runs selected original 6502 routines (from the public disassembly) against the simulator.
