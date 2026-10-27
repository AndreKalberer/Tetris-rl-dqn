# modern-rules: the Offline Sprint game

This project's own modern-rules Tetris (7-bag, hold, five-piece preview, SRS rotation, lock
delay) and the bot that plays it with timed, replay-verified inputs. Full write-up:
[docs/OFFLINE-SPRINT.md](docs/OFFLINE-SPRINT.md). Ruleset: [offline/RULES.md](offline/RULES.md).

| Path | Contents |
|---|---|
| `offline/` | Game engine, bot controller, player/recorder, 40-line acceptance evaluator, endurance runner. |
| `tests/` | Tests for the game and the endurance runner. |
| `docs/` | The Offline Sprint write-up and the October 2026 checkpoint notes. |
| `evidence/` | Acceptance, endurance and 100-hour endurance results. |
| `tools/` | `package_checkpoint.py`, which packaged the 2026-10-21 checkpoint. |
| `setup.ps1`, `play.ps1`, `evaluate.ps1`, `requirements*.txt` | The shared `.venv` (used by this folder and `old-dqn/`) and entry points. |

The bot scores boards with the learned evaluator in [`../old-dqn/`](../old-dqn/README.md), so
`old-dqn` must be on the import path too (the launchers and `pytest.ini` handle this).

```powershell
.\modern-rules\setup.ps1
.\modern-rules\play.ps1
.\modern-rules\evaluate.ps1
```
