# Modern offline checkpoint before NES work

Snapshot: `checkpoint/modern-offline-2026-10-21`.

This preserves the finished modern-style offline game and research implementation
before NES-specific development. NES has only been discussed; no NES engine or
NES-trained model is included.

## Retained results and limitations

- Offline Sprint: 20/20 reserved forty-line games passed, with exact timed-input
  replay, zero rejected inputs, board mismatches or planning overruns.
- Research: three independent training seeds; matched 60k snapshots survived the
  5,000-placement cap in 10/15 cases without search and 15/15 with beam-4 search.
  All five retained snapshots survived 25/25 searched cases. These are censored
  results. The seed2-100k one-ply regression is retained, not discarded.
- Offline endurance: seed 92000 reached 38,666 placements / 15,464 lines and was
  alive at the resource cutoff. Both segments replayed exactly. The first segment
  stopped from a reporting memory error at 1,900 placements / 757 lines;
  reporting was fixed and the same game continued. This is a lower bound from
  one stream, not an average, historical multiplier, or proof of infinite play.
- The rules are our documented modern offline ruleset: seven-bag, hold, next-five,
  SRS-style rotations, fixed 2-row/sec gravity. This is not a NES result or an
  official Guideline certification.

## What is saved

Git retains source, tests, pinned dependencies, five small inference policies,
selected evidence JSON, these notes, and milestone history. Full training
checkpoints and per-input endurance traces remain outside Git and are preserved
in the portable checkpoint ZIP with a SHA-256 file manifest. The archive also
contains a Git bundle for restoring history offline.

No virtual environment, caches, credentials, or temporary development files are
included. The earlier frozen Offline Sprint v1 ZIP remains unchanged.

## Restore and use

Extract the new checkpoint ZIP, enter its project directory, then:

```powershell
.\setup.ps1
.\play.ps1
.\evaluate.ps1
```

Verify the retained endurance game or continue the exact saved position:

```powershell
.\.venv\Scripts\python.exe -m offline.endurance --output runs/offline-endurance-v1/seed92000-resumed --verify-only
.\.venv\Scripts\python.exe -m offline.endurance --resume runs/offline-endurance-v1/seed92000-resumed/state.pkl --wall-seconds 1800 --output runs/endurance-next
```

Run from the extracted project root, with both preserved segments adjacent.
The replay resolver supports moved directories while leaving original protocols
intact. Resume states use Python pickle; load only trusted local checkpoint files.
Each continuation must use a fresh output directory.

Continue training toward an absolute total of 120k placements:

```powershell
.\.venv\Scripts\python.exe -m dqn.train_candidates --resume checkpoints/seed1-100000.pt --steps 120000 --output runs/continued-training
```

Full training checkpoints retain replay, optimizer, RNG and game state. The
five small `models/*.pt` files are inference-only exports.

Restore Git history separately after extracting `repository.bundle`:

```powershell
git clone repository.bundle restored-repository
git -C restored-repository switch --detach checkpoint/modern-offline-2026-10-21
```

The cloned repository contains source and policy exports. Copy `checkpoints/` and
`runs/` from the extracted snapshot when training or endurance resume is needed.
