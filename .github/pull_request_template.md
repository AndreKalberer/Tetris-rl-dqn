## What this changes

<!-- One or two sentences: the goal, and which bot folder(s) it touches (nes-bot, modern-rules, old-dqn). -->

## Why

<!-- The problem or target this serves, e.g. "score per lap", "crash-zone survival", "cleanup". -->

## Results

<!-- Numbers that back the change. Say where each comes from:
     sim = simulator estimate (zone_sim / sim tests), real = MesenCE game with replay verification.
     A result counts as verified only when a real game replays exactly. -->

| Check | Before | After | Source |
|---|---|---|---|
|  |  |  | sim / real |

## Checklist

- [ ] `python -m pytest -q` passes from the repository root (say which tests skipped and why)
- [ ] If simulator behaviour changed on purpose: survival floors checked and goldens re-recorded with `python nes-bot/tools/sim_golden.py --write`
- [ ] New files are in the right bot folder (see CONTRIBUTING.md)
- [ ] Docs updated: the bot folder README for new files, `nes-bot/docs/RESULTS-255.md` for new verified games, the root README only for verified headline results
- [ ] Overlaps with other open PRs noted below

## Overlaps and merge order

<!-- Other open PRs touching the same files, and which should merge first. "None" is fine. -->
