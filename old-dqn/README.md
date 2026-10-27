# old-dqn: the learned (DQN) bots

The project's reinforcement-learning work, kept for reference and because the modern-rules
bot still uses its board evaluator.

| Path | Contents |
|---|---|
| `dqn/` | Double DQN board evaluator over candidate placements (`candidates.py`, `train_candidates.py`), lookahead, robustness study, line challenge, and the legacy Stable-Baselines3 scripts (`train_dqn_server.py`, `eval_lines.py`, `diagnose_policy_dqn.py`, `enjoy_local_dqn.py`). |
| `tetris_macro_env.py` | The original placement (one decision per piece) wrapper around Tetris Gymnasium. |
| `models/` | Small trained policies used by the Offline Sprint (`seed1-100000.pt` is the default). |
| `play_bot.py`, `record_gif.py`, `assets/` | Live playback and the GIF recorder, with the recorded GIF. |
| `research-evaluate.ps1` | Repeats the lookahead research study. |
| `tests/` | Tests for candidate enumeration and the line challenge. |
| `evidence/` | The DQN robustness study results. |

Results and history: [../modern-rules/docs/OFFLINE-SPRINT.md](../modern-rules/docs/OFFLINE-SPRINT.md).
It uses the shared `.venv` set up by `modern-rules/setup.ps1`.
