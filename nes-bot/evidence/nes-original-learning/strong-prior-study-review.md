# NES study review — October 22, 2026

Read-only review at approximately 3:13 PM America/Denver. No processes, models, source, schedules, or experiment results were changed by this review.

The three independent 10,000-decision training budgets **and final selected-policy runtime evaluations are complete**. This is supported by the combined `three-seed-summary.json`, both stage summaries, all three `selected-runtime-through-10000/results.json` files, and complete live status. Neither recorded worker 1928 nor supervisor 20264 was present at the check; neither training directory had a `RUNNING.lock`.

Declared selections remain unchanged:

- Training seed 0: 10,000-decision checkpoint. Reused starts 1001/1013 reached level 30 at 240/240 lines and 242,002/284,886 points.
- Training seed 1: 5,000-decision checkpoint. Reused starts reached level 30 at 240/241 lines and 257,999/238,655 points.
- Training seed 2: 0-decision checkpoint, the initial untrained residual model over the fixed prior. Reused starts reached level 30 at 240/240 lines and 231,218/285,114 points. Describe this honestly as the selected initial policy; three completed training runs do not imply three improved learned policies.

All six selected-runtime episodes have exact full-RAM/RGB replay verification, no reported controller/board audit failures, and no runtime overruns. These starts were repeatedly used for checkpoint selection, so this is **selection evidence, not independent test evidence**. Each training seed completed 2,245 updates.

Selected `best.pt` SHA256 values recorded before any proposed fresh pilot:

- Seed 0: `9cebc8cbb5950c05bd4235ae0ae69a3e45b1e5989231c2f27df009292c3fa3f3`
- Seed 1: `ebeb025f40fd943abb124f5c6dc1a07d2dd235c1cfacf2a4bbae43d4ee128910`
- Seed 2: `98975ea2ddc782d06a5a18cc7dfd478b6bf8ea9cfde93e5a2ef092038d9d214c`

The declared rank uses success count, **uncapped median raw lines**, then mean score. This can reward a final clear overshooting the 240-line target by one line. Seed 1's selected rank `(2, 240.5, 248327)` beat its initial prior `(2, 240, 258166)` despite lower mean score, because 241 lines outranked 240. Preserve this declared selection and report the artifact; do not retrospectively change the ranking. A future separately declared study can cap lines at 240 before the score tie-break.

The completed Mesen transfer pilot used only selected training seed 0, on menu waits 1701/1713. The fixed-prior baseline cleared both at 240 lines (226,421/240,189 points); selected seed 0 cleared one at 240 lines/263,040 points and topped out in the other at 92 lines/81,569 points. All four inputs replayed exactly. The failed learned run incurred one planning overrun and five charged idle frames; this pilot cannot separate policy differences from that timing event. It establishes neither reliable RL improvement nor the broader acceptance target.

Recommended next bounded pilot, after the root agent's process/CPU coordination:

1. Freeze all three selected hashes, source revision, original image hash, emulator build/profile, configuration, and new output paths before the first start. Retain all failed/censored/error episodes and their input proofs.
2. Predeclare menu waits **1801, 1813, 1827**, start level 18, stop after clearing level 29/reaching level 30. Compare one identical fixed prior with all three preserved selections: 12 games per timing mode. These are development pilot starts, never the reserved 20 acceptance starts.
3. Use the tested `nes_original.compare` interface with both completed jobs, `--workers 2 --worker-cpu-masks 1 256`, and the isolated JIT dependency path. Prefer Mesen for original-game fidelity. Keep the modern worker's allocation separate.
4. Run **fixed logical reaction** (`--fixed-reaction`) and **charged wall-clock latency** as distinct, clearly labeled protocols/output directories. Do not pool their outcomes. Fixed timing addresses policy choice; charged timing measures end-to-end deployment on the measured host. Later RNG streams can diverge when different inputs alter timing; these are paired starting states, not identical future-piece tapes.
5. Report each seed separately, success counts, score, capped and raw lines, overruns/charged frames, and exact replay/audit status. Do not reselect checkpoints after viewing these starts. A three-start pilot is too small for acceptance or a reliable improvement claim.

Duplicate/race guards inspected: comparison refuses an existing output directory, any supplied training `RUNNING.lock`, duplicate training seeds, and inconsistent prior/pulse profiles. It records selected model hashes before gameplay and rechecks before model evaluation; parallel workers also verify their expected model hash. Creating a fresh output directory without `exist_ok` prevents two simultaneous launches from sharing that path. These guards do not prevent a different output path from duplicating the same experiment, so the root agent should launch exactly one coordinator per declared timing protocol and confirm active command lines first. Completed study checks above are a point-in-time observation, not permission for another agent to launch a job.

The reserved 20 games remain untouched by this proposed pilot. No 500,000 median-score achievement, reliable RL gain, level-255 crash coverage, or rebirth certification is established.
