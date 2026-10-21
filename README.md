# Tetris-rl-dqn

RL agents for Tetris using `tetris_gymnasium`, trained with **DQN + macro (placement) actions**.

Each "timestep" = 1 placed piece: the agent picks `(rotation, x, hold?)`, the wrapper teleports the piece and hard-drops it.

![Bot playing](assets/tetris_bot.gif)

## Project overview

This project explores a practical RL design choice: make one decision per placed piece instead of learning dozens of frame-level inputs. The macro-action wrapper, training scripts, checkpoint playback, and policy diagnostics are implemented in Python using Gymnasium, Stable-Baselines3, and PyTorch.

The GIF demonstrates checkpoint playback. It is an illustrative episode, not a benchmark or a representative evaluation sample.

## How it works

Standard DQN struggles on Tetris because per-frame control (left/right/rotate/drop) makes
rewards extremely sparse and credit assignment hard. This project reframes the problem with a
**macro-action wrapper** (`tetris_macro_env.py`): the action space becomes the set of legal final
placements `(rotation, x, hold?)`, and the wrapper teleports + hard-drops the piece. The agent
makes one decision per piece instead of dozens per piece, which makes the DQN converge far more
reliably. Built on `tetris_gymnasium` + Stable-Baselines3 (PyTorch).

## Setup

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
## Play (simple)
Plays the latest checkpoint in `models/` and auto-detects obs/hold/next-n/reward from the filename:
```powershell
python .\play_bot.py
```
Override with flags if needed.

Use a specific checkpoint or override settings:
```powershell
python .\play_bot.py --model-path .\models\<checkpoint>.zip --obs board --hold-actions --next-n 1 --reward-profile tetris
```

## Train (local)
```powershell
python .\dqn\train_dqn_server.py --timesteps 500000
```

With hold actions + next-5 lookahead:
```powershell
python .\dqn\train_dqn_server.py --timesteps 500000 --obs board --hold-actions --next-n 5 --reward-profile tetris
```

## Resume training
```powershell
python .\dqn\train_dqn_server.py --timesteps 2000000 --resume-from .\models\<checkpoint>.zip
```

## Watch the agent play locally
Pull the latest checkpoint from the server and render:
```powershell
.\dqn\peek_dqn.ps1 -Prefix 'dqn_tetris_board_hold' -Obs board -HoldActions 1 -NextN 1 -RewardProfile tetris
```

Or run directly with a local model:
```powershell
python .\dqn\enjoy_local_dqn.py --model-path .\models\<checkpoint>.zip --obs board --hold-actions --next-n 1 --reward-profile tetris --episodes 5 --sleep 0.05
```
## Create a GIF for GitHub
Records a short GIF (default output: `assets/tetris_bot.gif`):
```powershell
python .\record_gif.py --model-path .\models\<checkpoint>.zip --episodes 3 --max-steps 500 --fps 30
```
Auto-detects obs/hold/next-n/reward from the checkpoint filename unless you override them.

Loop forever and keep only the best episode (by lines cleared):
```powershell
python .\record_gif.py --episodes 8 --top-k 1 --select-metric lines --loop 0
```

If your checkpoint was trained without hold actions:
```powershell
python .\record_gif.py --model-path .\models\<checkpoint>.zip --no-hold-actions
```

## Diagnose / evaluate
```powershell
python .\dqn\diagnose_policy_dqn.py --obs board --hold-actions --next-n 1 --reward-profile tetris --n-steps 20000
```

Quick line-clear eval:
```powershell
python .\dqn\eval_lines.py --obs board --hold-actions --next-n 1 --reward-profile tetris --steps 20000
```

## TensorBoard
```powershell
ssh -L 16006:localhost:6006 user@your-server.example.com
# On the server:
tensorboard --logdir /root/tetris-rl-dqn/logs --port 6006 --host 127.0.0.1
```
Then open: http://localhost:16006

## Best models (as of Jan 2026)

These are historical reported results. The repository does not include the original evaluation logs, a random-policy baseline, or the trained checkpoint files; the numbers below have not been independently reproduced. They should not be read as a controlled comparison between reward profiles.

- `dqn_tetris_board_hold_32900000` — 75.3 lines/ep (obs=board, hold, next-n=1, tetris reward)
- `dqn_tetris_board_hold_32850000` — 65.0 lines/ep (obs=board, hold, next-n=1, tetris reward)
- `dqn_tetris_board_hold_next5_survival_56000000` — 34.8 lines/ep (obs=board, hold, next-n=5, survival reward)

## Reproducing a policy evaluation

Use an explicit checkpoint rather than whichever file happens to be newest. After setup, place your trained checkpoint in `models/` and run from the repository root, substituting its actual filename:

```sh
python dqn/diagnose_policy_dqn.py --model-path models/YOUR_CHECKPOINT.zip --obs board --hold-actions --next-n 1 --reward-profile tetris --n-steps 20000 --seed 42 --summary
```

Match `--obs`, hold actions, and `--next-n` to the training configuration. Repeat with seeds 43, 44, 45, and 46, keeping the checkpoint and step budget fixed. Record the commit, checkpoint SHA-256, dependency versions, hardware, seed, completed episode count, total lines, invalid-action rate, and reward profile alongside the raw output.

The diagnostic uses deterministic model predictions and resets successive episodes with `seed + episode_number`. Its fixed placement budget may stop mid-episode, and its reported `lines_per_ep` includes lines from that unfinished episode in the numerator. Use it as a policy diagnostic; do not describe it as a completed-episode benchmark. If no episodes finish, that ratio is not an episode average.

A controlled comparison should separately evaluate completed episodes for a random policy, the DQN policy, and any alternative action representation using the same seeds and environment rules. Report sample counts and variation, and keep each reward/observation configuration distinct. That comparison is not yet available; no new training or benchmark run is claimed here.

## Notes
- On PowerShell, avoid `&&` inside remote command strings; use `;` instead.
- For stable server performance, set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.

## License

[MIT](LICENSE)
