# Original NES level255 development

The current goal is to reach255 from an ordinary NTSC A-Type level18 start.
This requires genuine original game execution and a complete fresh-core proof,
not an injected level or a provisional simulator score. Rebirth after clearing
255 is a separate event.

```text
python -m nes_original.endurance --output NEW_DIRECTORY --mesen PATH_TO_MESEN --jit-deps PATH_TO_OPTIONAL_NUMBA --target-level 255 --seeds 33
```

Planner and emulator
default to CPU masks32/128; use available, separate masks on another machine.
Defaults use the retained lookahead prior. `--selector dellacherie` selects an
isolated experimental six-feature one-piece prior. `--selector risk-aware`
selects an isolated current/known-preview/seven-hypothetical-type search. Neither
experimental selector has an established native survival gain. All variants
charge measured planning latency, use original menus/controller input, and keep
the actual game/physics authoritative. No unseen RNG or future queue is queried.

Each attempt is bounded by an explicit gameplay-frame cap and emulator-process
deadline. A cap is censoring, not winning. The default frame cap is2million and
the per-process deadline6hours. These support long attempts; completing that
compute budget cannot meet the goal.

`learn.evaluate(..., target_level=255)` retains the default30 for earlier work.
It audits each settled piece with `MarathonProgress`, rejects skipped boundaries
or inconsistent counters, and requires3290..3293 first-cycle lines for genuine
255 from18. An exact replay can reproduce corruption, so raw255 alone cannot
certify success. Audited target evidence must also pass fresh full RAM/RGB and
CPU/PPU clock reconstruction. Every gameplay result and controller statistic is
written before replay verification, then marked verified only after it passes.

The endurance wrapper additionally saves full controller-prefix binary journals
and cached RAM/clock metadata every600frames and on faults. Intermediate cached
observations are explicitly uncertified. A prefix journal is recovery evidence,
not a fresh full proof or a controller-state checkpoint. Fault journals retain
raw RAM even when corrupted counters cannot be decoded. Source, policy,
controller profile, core and ROM identity must remain recorded with each run.

`--selector dellacherie-alerts` prioritizes proposals outside four documented
conditional crash signatures. It is not a complete safe-action map. Logs retain
alerts and forced fallbacks. `--no-softdrop-levels 159` enables a separate timing
experiment only at that raw level, retaining the ordinary soft/fall-byte guards.
Neither option guarantees that the original ROM will avoid a crash.

`--transport pipe-blocking --verification-transport pipe` opts into faster,
watchdog-bounded gameplay IPC with a separate fresh reconstruction through the
original polling transport. The default remains polling. Seven Windows IPC
safety cases and paired native proofs passed before this option was exposed;
one480-frame benchmark measured1.58x throughput, not stronger play. The bridge,
ROM, controller validity and emulated CPU/PPU timing stay authoritative.
`--stop-on-success` ends a development suite after its first fully verified
target; all earlier failures remain in the result list.

Keep old simulator pilots, failed native runs, stopped-at30 trials and frozen
selection studies intact. Development endurance on known starts supplies no
untouched20-game acceptance result and no learned-improvement claim. The original
late score/NMI crashes and long counter regimes remain part of the unmodified
game; none is removed or silently replaced by a safer ruleset here.
