# Modern endurance: absolute100-hour resource stop

The user requested100total simulated hours and a four-core allowance. Deployed
worker25964 resumes the same checkpointed game, stops before tick21,600,001,
and saves game/controller state at21,600,000frames (60FPS). This is a censored
resource stop, not a top-out, win or uncensored survival estimate.

Source snapshots here are audit copies; deployed location is the workspace's
work/overnight-modern/worker_v3.py, with its preserved source/, source-manifest.json
and trusted continuation state. Keep the script at that location to run it.
Thirteen invalid CLI cases and exact+30/+60/zero-extra-tick resumption tests passed;
segmented and uninterrupted game/controller continuation matched. No game counter,
physics, model or piece-stream edits. The launchers live in workspace outputs/.

Full-planner24samples/setting on3cloned states:2threads median23.315ms,
4threads23.340ms. No observed4-thread gain; selected actions matched. Production
allows logical2/4/6/10, CPUbudget4, Normal priority, and retains2Torchthreads.
This does not promise a4x speedup. No live source was altered during tests.
