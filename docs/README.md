# Documentation

Start with the [project README](../README.md). This page lists every document by topic.

## NES Tetris on the original cartridge (current work)

| Document | What it covers |
|---|---|
| [NES-255.md](NES-255.md) | How the level-255 bot works, why level 255 is hard, how to run, watch and test it, known limits. |
| [RESULTS-255.md](RESULTS-255.md) | Every verified level-255 and rebirth game (bot version, seed, lines, uncapped score) and where games were lost. |
| [NES-ORIGINAL-REFERENCE.md](NES-ORIGINAL-REFERENCE.md) | The MesenCE 2.2.1 reference path: emulator setup, game image check, play / replay / train / compare commands. |
| [../tools/README.md](../tools/README.md) | Launcher, batch runner, crash-zone simulator, tuners and replay viewer. |
| [../nes_original/README.md](../nes_original/README.md) | The original-game environment adapter (nes-py core) and its fidelity limits. |
| [../nes_original/ENDURANCE.md](../nes_original/ENDURANCE.md) | Development notes on long original-game runs toward level 255. |
| [../nes_original/LOOKAHEAD.md](../nes_original/LOOKAHEAD.md) | Faster legal controls and visible-next search (first level-29 clear). |
| [../nes_original/TRAINING.md](../nes_original/TRAINING.md) | Original-game level-29 training experiments. |

## Earlier NES simulator work

| Document | What it covers |
|---|---|
| [../nes/README.md](../nes/README.md) | A separate, provisional NES A-Type simulator (`.\play-nes.ps1`, `.\play-nes.ps1 -Bot -StartLevel 18`). Not an emulator; whole-ROM parity was never demonstrated. |
| [../nes/RULES.md](../nes/RULES.md) | The simulator's mechanics, sources and fidelity gaps. |
| [../nes/TRAINING.md](../nes/TRAINING.md) | A one-seed, simulator-trained learning pilot. Not an original-ROM result. |

The nes-py core was later found to keep executing after a CPU JAM instruction, so late-game
crash and rebirth results are only claimed from the MesenCE reference path.

## Offline Sprint (modern-rules game)

| Document | What it covers |
|---|---|
| [OFFLINE-SPRINT.md](OFFLINE-SPRINT.md) | Setup, play, the 20/20 acceptance result, the lookahead research study, endurance run, training, legacy SB3 scripts. |
| [../offline/RULES.md](../offline/RULES.md) | The frozen Offline Sprint v1 ruleset. |
| [CHECKPOINT.md](CHECKPOINT.md) | The modern offline checkpoint taken before NES work, and how to restore it. |

## History and evidence

- [../progress.md](../progress.md) — the dated, unedited development log (newest first). It
  stays at the repository root because some experiment scripts update it in place.
- [../evidence/](../evidence/) — retained result files, protocols and audits from earlier
  milestones (offline acceptance, robustness, NES reference and crash-model research tables).
