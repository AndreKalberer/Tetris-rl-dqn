# Documentation

Start with the [project README](../README.md). This page lists every document by topic.
Each bot keeps its documents in its own folder: `nes-bot/`, `modern-rules/` and `old-dqn/`.

## NES Tetris on the original cartridge (current work)

| Document | What it covers |
|---|---|
| [NES-255.md](../nes-bot/docs/NES-255.md) | How the level-255 bot works, why level 255 is hard, how to run, watch and test it, known limits. |
| [RESULTS-255.md](../nes-bot/docs/RESULTS-255.md) | Every verified level-255 and rebirth game (bot version, seed, lines, uncapped score) and where games were lost. |
| [NES-ORIGINAL-REFERENCE.md](../nes-bot/docs/NES-ORIGINAL-REFERENCE.md) | The MesenCE 2.2.1 reference path: emulator setup, game image check, play / replay / train / compare commands. |
| [nes-bot/tools/README.md](../nes-bot/tools/README.md) | Launcher, batch runner, crash-zone simulator, tuners and replay viewer. |
| [nes-bot/nes_original/README.md](../nes-bot/nes_original/README.md) | The original-game environment adapter (nes-py core) and its fidelity limits. |
| [nes-bot/nes_original/ENDURANCE.md](../nes-bot/nes_original/ENDURANCE.md) | Development notes on long original-game runs toward level 255. |
| [nes-bot/nes_original/LOOKAHEAD.md](../nes-bot/nes_original/LOOKAHEAD.md) | Faster legal controls and visible-next search (first level-29 clear). |
| [nes-bot/nes_original/TRAINING.md](../nes-bot/nes_original/TRAINING.md) | Original-game level-29 training experiments. |

## Earlier NES simulator work

| Document | What it covers |
|---|---|
| [nes-bot/nes/README.md](../nes-bot/nes/README.md) | A separate, provisional NES A-Type simulator (`.\nes-bot\play-nes.ps1`, `.\nes-bot\play-nes.ps1 -Bot -StartLevel 18`). Not an emulator; whole-ROM parity was never demonstrated. |
| [nes-bot/nes/RULES.md](../nes-bot/nes/RULES.md) | The simulator's mechanics, sources and fidelity gaps. |
| [nes-bot/nes/TRAINING.md](../nes-bot/nes/TRAINING.md) | A one-seed, simulator-trained learning pilot. Not an original-ROM result. |

The nes-py core was later found to keep executing after a CPU JAM instruction, so late-game
crash and rebirth results are only claimed from the MesenCE reference path.

## Offline Sprint (modern-rules game)

| Document | What it covers |
|---|---|
| [OFFLINE-SPRINT.md](../modern-rules/docs/OFFLINE-SPRINT.md) | Setup, play, the 20/20 acceptance result, the lookahead research study, endurance run, training, legacy SB3 scripts. |
| [modern-rules/offline/RULES.md](../modern-rules/offline/RULES.md) | The frozen Offline Sprint v1 ruleset. |
| [CHECKPOINT.md](../modern-rules/docs/CHECKPOINT.md) | The modern offline checkpoint taken before NES work, and how to restore it. |

## History and evidence

- [../progress.md](../progress.md) — the dated, unedited development log (newest first). It
  stays at the repository root because some experiment scripts update it in place.
- Retained result files, protocols and audits now sit with the bot they belong to:
  [nes-bot/evidence/](../nes-bot/evidence/) (NES reference, crash-model research tables, learning
  pilots), [modern-rules/evidence/](../modern-rules/evidence/) (offline acceptance, endurance) and
  [old-dqn/evidence/](../old-dqn/evidence/) (DQN robustness study).
