# NES late-game and rebirth readiness

Audit date: October 22, 2026. Workspace: `C:\Users\andre\Documents\Codex\2026-10-21\ex`.

**The current setup is useful for original-ROM bot development, but it is not ready to certify a hardware-faithful level-255→0 run.** We now have a concrete CPU crash-behavior difference in the installed `nes-py 9.0.1` core, confirmed against an independent portable MesenCE build. This does not erase the original-game runs or their exact within-core replays. It limits the hardware-equivalence claim, especially for late-game crashes.

## Confirmed diagnostic result

I generated three tiny, self-authored NROM cartridges. They contain no Nintendo program or artwork. Each initializes RAM markers using ordinary CPU instructions; the Python observer never patches RAM. The cases are a jump-to-self control, a NOP control, and the CPU JAM/KIL opcode `$02` followed by marker-writing code that should be unreachable after a real JAM.

- `nes-py 9.0.1`: jump-to-self leaves the post-marker and counter at zero; NOP executes them; **JAM also executes them**. The post-marker becomes `$5A` and the counter advances across three API steps.
- Official portable **MesenCE 2.2.1**: jump-to-self and JAM both leave the post-marker/counter at zero; NOP executes them. All three headless probes exited cleanly after their observation window.

The independent core's opcode table maps `$02` to its halt handler, which prevents further normal instruction execution and interrupts. [Mesen CPU implementation](https://github.com/nesdev-org/MesenCE/blob/2.2.1/Core/NES/NesCpu.cpp)

Evidence is in `work/nes-late-fidelity/diagnose_kil.py`, `kil-diagnostic.json`, `mesence-check.lua`, and `mesence-*-result.json`. The initial diagnostic revision is retained separately; the final padded probe prevents incorrect operand-length decoding from obscuring the marker result. This is a CPU diagnostic, **not a Tetris score or a legal gameplay run**. It proves a failure for this class of CPU halt; it does not prove that every possible Tetris crash is suppressed.

## Why this matters for original Tetris

At high levels, the original scoring routine can extend game processing across an NMI. Shared scratch RAM can then corrupt jump-table dispatch and send execution into unintended instructions. Original technical research places the earliest NTSC possibility at the transition to level 155. Crash outcomes depend on execution timing and the particular path, so a core that mishandles halting instructions cannot establish authentic crash avoidance merely by continuing to run. [Original technical explanation](https://negative-seven.github.io/tetris_explained/#high-levels--crashing-the-game)

The SHA-256-verified official `nes-py 9.0.1` source also shows a fixed **29,781 CPU cycles per API step**, with three PPU cycles per CPU cycle, and instruction batching for mappers without CPU-cycle hooks. Thus one API step should not be equated unconditionally with the next actual video-frame boundary. These are timing-contract limitations to investigate, not proof that all ordinary gameplay is wrong. The source archive used was the [official 9.0.1 source distribution](https://files.pythonhosted.org/packages/c0/77/d27858362aaaea8a371062515ba5c4ae0c48dc482797998485c998c70e8c/nes_py-9.0.1.tar.gz), SHA-256 `f9aafc42a90600af39e844b6c0a153fdfc2b7635997cb70be345c22e58b4557b`.

An additional source-level longevity concern is its signed `int` CPU-cycle accumulator. A normal 32-bit signed accumulator crosses its range after roughly 72,110 such steps, around 20 minutes of nominal game time. Only parity is visibly used in this source, so this is **not an observed failure**; signed overflow still warrants a long-duration test or correction before multi-hour claims.

## Original counters and progression

The original program increments an eight-bit level value. A real increment from 255 wraps to 0, restoring level-0 gravity. The line counter is separate: its low byte contains two decimal digits and its high byte is binary hundreds; its raw numeric range is 0–25,599. At the next line it wraps to zero. Neither on-screen glyphs nor the level byte alone represent total lifetime progress. The score stored by the original game caps at 999,999. [Pinned original disassembly](https://github.com/CelestialAmber/TetrisNESDisasm/blob/94581c0306fa0009fe11fde48222875549d15b8b/main.asm)

I assembled the pinned `incrementLines` routine in scratch memory under py65 and compared 1,000 randomized counter/level/clear-count cases with a pure arithmetic projection; all matched. This was source-level arithmetic validation, **not a whole-emulator or human-play result**. From an 18 start, the calculation gives:

- 130 total lines: level 19; 240: level 30.
- Level 235 begins at 2,290 lines and ends at 3,100: **810 lines in that level**.
- Level 255 begins at 3,290; the first 255→0 increment occurs at 3,300.
- In the following cycle, the 810-line level is **189**, spanning 5,190–5,999 total lines. Cycles cannot be represented by simply repeating the first-cycle schedule.
- At absolute line 25,600, the source arithmetic's raw line counter becomes zero; this is a different event from a level rebirth.

The comparison uses the sign bit of an eight-bit level-minus-threshold result, rather than an ordinary greater-than comparison. `work/nes-late-fidelity/progression_audit.py` and `progression-audit.json` retain the source hashes, cases and trajectory results. These independently confirm the parent agent's proposed exact progression helper.

## Gaps in our surrounding tools

The following are project-code findings, separate from emulator hardware fidelity:

1. Current training/evaluation and lookahead deliberately finish at level 30. Their frame, wall-time and placement limits are development limits, not an endurance/rebirth protocol.
2. **Fixed during this audit:** the new original-game lookahead now uses `nes_original.progression.after_clear` from current raw level/line counters, replacing the linear simulator projection. Its source arithmetic matched this audit's 1,000 cases; dedicated parent-agent tests cover the long level, rebirth and separate line-counter wrap. The frozen older learner and generic simulator-derived neural predicted-level feature still use linear context and remain unsuitable for a 255/rebirth claim.
3. `OriginalGame.observe` correctly decodes binary hundreds, but exposes only raw level and raw lines. The new `MarathonProgress` helper now tracks monotonic progress and validates counter changes; callers must integrate it into long-run rewards and occupancy accounting. Existing raw-line subtraction would break at the 25,600-line counter wrap.
4. Neural features normalized around levels 18–29 are outside their training distribution at 138/255 and reset sharply at rebirth. Source-correct counters do not make the current policy trained for those states.
5. `over` detects the normal game-over phase or leaving gameplay mode. A crashed CPU can remain inside a nominal gameplay state. Add a separately classified semantic-progress watchdog, bounded error logs, raw diagnostic capture and an explicit `suspected_crash` outcome; never count this as an ordinary top-out or silently restart it.
6. Native snapshots are useful in memory. Portable nes-py saves currently reconstruct the entire episode from boot/input history. That becomes increasingly slow for long-run recovery and curriculum starts. Mesen's binary native savestates offer a reference-path option, but are core/build-specific and cannot consume an opaque nes-py snapshot.
7. HUD interpretation is already unreliable at level 30, before the later palette glitches. The project's verified level-30/240-line proof displays **LEVEL 00** because the original decimal-display lookup has only 30 entries; index 30 reads the following table's `$00`. This is **not** level-255 rebirth. Read validated game counters instead of displayed decimal digits. See `outputs/nes-level29-success.json` and `outputs/nes-level29-cleared.png`; the original lookup is at [the pinned render/table code](https://github.com/CelestialAmber/TetrisNESDisasm/blob/94581c0306fa0009fe11fde48222875549d15b8b/main.asm#L2584). A RAM-state bot is less affected by colors than a vision-only policy, but the core must still execute the original rendering/timing code faithfully.

## Reference backend that now works locally

Use **MesenCE**, the current community successor, as an independent reference candidate. The old `SourMesen/Mesen2` repository is archived and points to it. The verified stable portable Windows release here is [MesenCE 2.2.1](https://github.com/nesdev-org/MesenCE/releases/tag/2.2.1). Its downloaded ZIP SHA-256 is `c19e10b3b3865eb0e86e4d01bf09a5027a4452081c04b5bd5b6cbbf76e346fd6`.

Executable: `work/nes-late-fidelity/mesence-2.2.1/Mesen.exe`. A local `settings.json` containing `{}` beside the executable selects portable configuration and avoids the first-run wizard. All launched probes were hidden, on one spare logical CPU at BelowNormal priority. No global install or live training environment change was made.

Verified command shape:

```powershell
Mesen.exe --testRunner --timeout=10 --enableStdout --doNotSaveSettings --debug.scriptwindow.allowioosaccess=true probe.lua authored-probe.nes
```

The trusted probe uses I/O only to write its scratch results. `emu.log` did not provide those Lua records on redirected stdout in this run, so the scratch JSON is the evidence. The official runner has a bounded timeout and accepts one ROM plus Lua scripts. [Runner implementation](https://github.com/nesdev-org/MesenCE/blob/2.2.1/UI/Utilities/TestRunner.cs)

The Lua API provides `inputPolled` callbacks and `emu.setInput`, physical NES RAM reads, CPU/PPU state inspection, and binary `createSavestate`/`loadSavestate`; the savestate calls are restricted to execution-memory callbacks. The new reference adapter retains our controller/observation boundary using these APIs. [Official API definitions](https://github.com/nesdev-org/MesenCE/blob/2.2.1/UI/Debugger/Documentation/LuaDocumentation.json)

## Independent original-ROM driver smoke: passed

I also ran the existing private USA game image **in place** through MesenCE, without copying it or altering any game RAM. Its reported SHA-1 matched `77747840541bfc62a28a5957692a98c550bd6b2b`. Normal Start/Right/A+Start input reached the genuine level-18 game at video frame 342. The bounded trace then recorded 180 gameplay frames.

Across 522 observed video frames, there were 521 controller-input callbacks: exactly one per frame after the initial startup frame. ROM frame-counter deltas after startup and excluding the deliberate restore were all one across 177 samples. The input callback exposed PPU scanline 241/cycle 0 through `emu.getState`; available flat keys include `cpu.pc`, `cpu.cycleCount`, `frameCount`, `ppu.frameCount`, `ppu.scanline`, and `ppu.cycle`.

At the original NMI entry `$8005`, an execution-memory callback created a 7,110-byte in-memory native savestate. Loading it succeeded, restored all 2 KiB of RAM immediately, and reproduced the next 30-NMI continuation's full RAM exactly with the same Down inputs. This verifies basic driver and restore capability; it is **not an uninterrupted gameplay score, a whole-RGB restore proof, or late-game crash validation**.

The script and evidence are `work/nes-late-fidelity/mesence-original-smoke.lua`, `mesence-original-smoke.json`, `mesence-original-trace.csv`, `mesence-original-state-keys.tsv`, and `mesence-original-smoke.png`. Only the reference snapshot API changed emulator state; the script performed zero direct game RAM writes. The live training worker, its environment, and the modern endurance worker were not changed or monitored. `audit-manifest.json` pins the diagnostic files and downloads.

## Required validation before a rebirth claim

The authored CPU comparison, original-ROM menu/control/restore smoke and bounded closed-loop bot driver are complete. Next run timing-sensitive CPU/PPU/NMI tests in the selected reference build and longer controller trials. Test 137→138, 154→155, long-level boundaries, 254→255→0 and post-rebirth speed changes with clearly labeled diagnostic states; injected-state tests must never be reported as full legal runs. Finally obtain uninterrupted original-game input proofs with no patches or injected state.

Cross-emulator replay must align power-on settings, CPU/PPU phase, controller polling and actual video-frame boundaries. The same raw nes-py step-indexed inputs need not be bit-identical in another core under different initial phase. Compare aligned game events and relevant RAM; keep full-RAM/RGB hashes for within-build reproducibility. Different RGB palettes alone are not a gameplay-fidelity failure.

No uninterrupted 255→0 Tetris game or crash-condition matrix has been validated, and no production training worker has been migrated. Existing original-game level-29 clears remain labeled nes-py development results. The MesenCE path now has a working, independently tested closed-loop driver; it is not a certification of whole-hardware equivalence.

## Reusable closed-loop reference adapter

Added only new project files `nes_original/mesen.py`, `nes_original/mesen_bridge.lua`, and `tests/test_mesen_reference.py`. `MesenGame` supplies genuine-menu `reset(start_level=18, seed=...)`, native-byte `step`, the existing public observation fields, normal-game-over detection, explicit RGB capture, input replay/save/load and `close`. `seed` remains a menu idle count. The ROM is read in place, hash checked, and never copied or patched.

The measured input contract differs from nes-py: the callback waits at PPU scanline 241/cycle 0 before accepting the current command. Right immediately changes x from 5 to 6 in the next returned observation; after release, Left immediately restores 5. The observation therefore declares `input_pipeline=0` and `pending=0`. The parent planner now honors that marker rather than projecting/draining an extra input frame.

Default IPC uses two private, randomly named Windows pipes. The tested `transport='file'` fallback uses sequence-checked atomic files with bounded retries for Windows reader handles. No screen data crosses the bridge each frame; only public-state decoding and diagnostic 2 KiB RAM/clock reads occur. Explicit final screen capture does not advance emulation. The pipe probe ran 180 real gameplay commands after the compulsory idle and reproduced the entire boot/input history in a fresh process. Full RAM, RGB, CPU cycle/PC and PPU/frame clocks matched exactly. File and pipe transports also produced the same final fingerprints in this controlled profile. The matched two-process probe took 6.35 seconds with pipes versus 23.16 seconds with files; these figures include process startup and captures, not just IPC latency.

Nine focused test cases have passed. The original eight passed together in 12.9 seconds; the added hard process-deadline test passed in 3.64 seconds. They cover real-menu input/replay, an idle host watchdog terminating only its owned CE child, a total wall deadline holding while commands remain active (including menu frames), intentional pipe disconnection unblocking Lua and returning exit code 2, observation privacy, replay identity rejection, and separate frozen-NMI/frozen-game/unexpected-mode watchdog outcomes. Unit watchdog arrays are standalone data, **not injected emulator RAM**. A suspected crash is an exception with retained diagnostic state, never an ordinary top-out or successful completion. The independent host watchdog is necessary because a blocking Lua C read cannot execute Lua timeout checks. Close retains bounded session artifacts and closes only owned pipes/processes.

Usage in the isolated existing environment:

```python
from nes_original.mesen import MesenGame
with MesenGame(
    executable=r'C:\Users\andre\Documents\Codex\2026-10-21\ex\work\nes-late-fidelity\mesence-2.2.1\Mesen.exe',
    session_root=r'C:\Users\andre\Documents\Codex\2026-10-21\ex\work\nes-late-fidelity\adapter-sessions',
) as game:
    observation = game.reset(start_level=18, seed=0)
    observation = game.step(128)  # genuine Right button
    game.save_replay('runs/reference-example.json')
```

The dedicated portable build must already contain its extracted native DLL/database and pristine local `settings.json` (`{}`). The adapter records executable, native-core DLL, ROM-database and Lua-bridge hashes plus explicit NTSC/power-on settings; mismatched replay identities are rejected. Its defaults reserve logical CPUs 12/14 at BelowNormal on this workstation; other machines must pass a suitable affinity mask or `None`. Windows-only startup, a 30-second host-action deadline and a one-hour process deadline are explicit current limits. Native binary snapshots remain deferred; portable continuation currently replays all inputs from boot. Raise action/process deadlines deliberately for longer analyses or episodes. No claim is made that an unaligned nes-py input trace will reproduce across cores.

Evidence: `work/nes-late-fidelity/reference-adapter-{pipe,file}-smoke.json`, matching `*-replay.json`, `reference_adapter_smoke.py`, retained per-process `adapter-sessions`, and opt-in tests using `NES_MESEN_EXE`. The adapter smoke made zero game RAM writes.

The existing adaptive `FeedbackController` with two-frame horizontal pulses also ran against CE from genuine level 18, menu-wait seed 1. It completed **20 placements, 7 lines and score 8,930** in 1,128 gameplay frames, remained alive, and reported zero target misses, predicted-board differences, occupancy-audit failures or piece-type differences. Reconstructing that whole adaptive run in a fresh CE process matched the complete RAM/RGB/CPU/PPU fingerprint exactly. This bounded controller-plus-replay test took 44.55 seconds. The controller's decision time was recorded but not charged to native frames, so this is an offline controller test, not a certified realtime match. See `reference-controller-smoke.json`, `reference-controller-smoke-replay.json`, and `reference_controller_smoke.py` in the same scratch directory.

## Subsequent full reference-core clear

The parent controller has now legally cleared level29 in MesenCE2.2.1 from18:
menu-wait33,240lines,rawlevel30,255566score,608placements,33591gameplayframes,
alive. Full current+visible-next search, two-frame horizontal pulses, normal
gravity and zero game RAM/ROM edits. Runtime planning was charged, with zero
overruns, target misses, predicted-board differences or occupancy failures.
Fresh-process boot/input replay matched full RAM/RGB/CPU/PPU exactly. Earlier
menu-wait17 ended181lines/level24 and remains retained. This is development
search/controller evidence, not a reserved reliability result or an RL gain.

Native learning integration is also now tested: actual reference-game transitions,
optimizer/RNG/replay continuation, exact next weights/input proof and parking the
training core during separate validation all passed. Full trainer continuation
rejects another core. The currently running three-seed study remains on its
immutable nes-py source; it has not been silently migrated or relabeled.

No255/rebirth/crash-matrix certification follows from this level30 milestone.
Evidence: evidence/nes-reference/first-clear-{result,replay}.json; opt-in
tests/test_mesen_learning.py. Optional compiled search pins are in
requirements-nes-search.txt; setup and commands in docs/NES-ORIGINAL-REFERENCE.md.
