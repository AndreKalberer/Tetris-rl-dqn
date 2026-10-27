# nes-bot: the original NES Tetris bot

The current bot. It plays Nintendo's original Tetris in MesenCE 2.2.1 with controller input
only, reaches level 255, survives the level-155+ crash zones and is being pushed for score
and rebirths. How it works: [docs/NES-255.md](docs/NES-255.md). Every verified game:
[docs/RESULTS-255.md](docs/RESULTS-255.md).

| Path | Contents |
|---|---|
| [`nes_original/`](nes_original/README.md) | The bot: MesenCE and nes-py adapters, Lua bridge, planner and evaluator (`sim.py`, `fast.py`), crash model, endurance runner, replay and verification. Tuned weights in `nes_original/weights/`. |
| [`nes/`](nes/README.md) | The NES rules engine (`engine.py`, `actions.py`) that `nes_original` builds on, plus the early simulator learning pilot. |
| [`tools/`](tools/README.md) | `run255.sh` / `run-score.sh` launchers, batch runner, crash-zone simulator, tuners, replay viewer. |
| `tests/` | Tests for this bot, including the simulator regression goldens in `tests/data/`. |
| `docs/` | Bot docs: how it works, results, the MesenCE reference setup. |
| `evidence/` | Retained NES result files, protocols and research tables. |
| `models/` | Small policies from the NES learning pilot. |
| `requirements-nes*.txt`, `setup-nes*.ps1`, `play-nes.ps1` | Environments and entry points. |

Run everything from the repository root, for example:

```bash
sh nes-bot/tools/run-score.sh 957 255
python nes-bot/tools/zone_sim.py --zone early --games 400 --output runs/zone-early.json
PYTHONPATH=nes-bot python -m nes_original.endurance --help   # or `pip install -e .` once
```

To freeze a bot version for a long experiment, copy this whole folder and run
`python -m nes_original.endurance ...` from inside the copy.
