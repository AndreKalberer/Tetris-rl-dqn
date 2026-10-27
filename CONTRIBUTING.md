# How work gets into this repository

This page is the house system for branches, pull requests, merges and documentation. It
applies to anyone (person or Claude) adding work.

## 1. Where new work goes

Each bot has one top-level folder. Put new files next to the code they belong to.

| You are adding | Put it in |
|---|---|
| NES bot logic (planner, evaluator, crash model, controller, endurance runner) | `nes-bot/nes_original/` |
| NES rules engine changes | `nes-bot/nes/` |
| Tuned weights for the NES bot | `nes-bot/nes_original/weights/` (one JSON per weight set, named by what it is for, e.g. `tetris-lowstack-r0.json`) |
| Launchers, simulators, tuners, batch runners for the NES bot | `nes-bot/tools/` (add a line to `nes-bot/tools/README.md`) |
| Modern-rules game or its bot | `modern-rules/offline/` |
| Learned (DQN) models and training | `old-dqn/` |
| Tests | the bot's own `tests/` folder (`nes-bot/tests/`, `modern-rules/tests/`, `old-dqn/tests/`); test file names must be unique across the repo |
| Docs about one bot | that bot's `docs/` folder or README |
| Small retained result files (protocols, summaries, research tables) | the bot's `evidence/<topic>/` folder |
| Docs about the whole project | the root `README.md` or `docs/` |

Never commit: `runs/` (game output, logs), game images (`*.nes`), large checkpoints
(`*.pt` except the small whitelisted policies), API keys or `.env` files. `runs/` lives at the
repository root and stays untracked.

Code that is shared between bots stays where it is and is imported (each bot folder is an
import root). Do not copy a module into a second folder.

## 2. Branches

- Never commit to `main` directly. One branch per goal, made from the latest `main`.
- Name: `<area>/<short-topic>`, lowercase with hyphens. Areas used so far:

| Area | For |
|---|---|
| `score/` | score and rebirth work on the NES bot |
| `nes/` | other NES bot features (evaluator, crash zones, controller) |
| `fix/` | bug fixes |
| `perf/` | speed-ups with no behaviour change |
| `tests/` | tests only |
| `docs/` | documentation only |
| `cloud/` | running on cloud machines (Linux, headless MesenCE) |
| `modern/` | the modern-rules game and bot |
| `dqn/` | learned bots |
| `repo/` | layout, tooling, CI, this file |

Examples: `score/third-piece-expectimax`, `fix/keep-late-plan`, `repo/bot-folders`.

## 3. Pull requests

- **One goal per PR.** If a PR needs a cleanup first, open the cleanup as its own PR.
- **Moves separate from edits.** When moving or renaming files, put the pure moves in one
  commit and the edits in another, so git can follow every file through a rebase.
- **Fill in the template** (`.github/pull_request_template.md`): what, why, results with their
  source, checklist, overlaps.
- **Results language.** Say "sim" for simulator estimates and "verified" only for real MesenCE
  games that replayed exactly. A goal counts as reached only after both.
- **Tests.** Run `python -m pytest -q` from the repository root before opening the PR. If
  simulator behaviour changes on purpose, check the survival floors first
  (`SIM_FULL_BASELINE=1`, about 10 minutes), then re-record the goldens with
  `python nes-bot/tools/sim_golden.py --write` in the same PR, and say so in the PR.
- **Draft PRs** are fine for work that still needs real-game checks; mark ready when done.

## 4. Merging

- Andre merges on GitHub with **"Create a merge commit"** (not squash or rebase), so each
  PR's commits, including file moves, stay intact in history.
- **Order when PRs overlap:** merge the one whose results are ready first. Every other open PR
  then updates from `main` (`git fetch origin && git rebase origin/main`, or merge `main` in)
  and re-runs the tests before it is merged.
- Delete the branch on GitHub after the merge (the button on the merged PR).
- After merging, everyone's local checkout updates with `git pull` on `main`.

## 5. Keeping the README and docs current

Update in the same PR as the change, not later:

| When | Update |
|---|---|
| New verified NES games (level 255, rebirth, score) | `nes-bot/docs/RESULTS-255.md` (bot version, seed, lines, uncapped score) |
| A new headline result is verified | the "Headline results" table in the root `README.md` |
| New file or folder in a bot | that bot folder's `README.md` table |
| New tool | `nes-bot/tools/README.md` |
| How the bot works changed | `nes-bot/docs/NES-255.md` |
| Any notable step | a dated line at the top of `progress.md` |

The root README keeps only verified results and the folder map; details go in the bot folders.

## 6. Pushing from Andre's PC

Cloud sessions without GitHub access hand over a patch file; Andre pushes it from his PC.
Use a separate worktree so the main checkout (and any paused runs in `runs/`) is never
switched to another branch:

```powershell
cd C:\Users\andre\Documents\Codex\2026-10-21\ex\work\Tetris-rl-dqn
git fetch origin
git worktree add ..\Tetris-rl-dqn-pr -b <area>/<topic> origin/main
cd ..\Tetris-rl-dqn-pr
git am -3 <path-to>\change.patch
git push -u origin <area>/<topic>
gh pr create --fill --base main
cd ..\Tetris-rl-dqn
git worktree remove ..\Tetris-rl-dqn-pr
```

After the PR is merged on GitHub, update the main checkout:

```powershell
cd C:\Users\andre\Documents\Codex\2026-10-21\ex\work\Tetris-rl-dqn
git pull origin main
```

## 7. Running code after the folder split

Each bot folder is an import root. The launchers and `pytest.ini` set the path themselves.
For hand-typed `python -m ...` commands, run once per Python environment, from the
repository root: `python -m pip install -e .` (or set `PYTHONPATH`, for example
`PYTHONPATH=nes-bot`). A frozen copy of the NES bot is just a copy of `nes-bot/`; run
`python -m nes_original.endurance ...` from inside the copy.

## 8. Updating a branch made before the folder split

Edits to files that moved follow the move automatically during `git rebase origin/main`.
Only files the branch *added* stop the rebase with a "CONFLICT (file location)" message; git
has already put them in the new folder, so:

1. Check where git put each new file. New tests in particular must go to the right bot's
   `tests/` folder (git may guess `nes-bot/tests/` for a modern-rules test): fix with `git mv`.
2. A new launcher copied from `run255.sh` needs the same three changes as the moved ones:
   `ROOT=$(cd "$(dirname "$0")/../.." && pwd)`, `W="$ROOT/nes-bot/nes_original/weights"`,
   and `export PYTHONPATH="$ROOT/nes-bot"` before `exec`.
3. `git add -A`, then `git rebase --continue`, then run the tests.

Tested on the open 50M score patch and the modern-rules port patch: both rebased with only
these steps and their tests pass.
