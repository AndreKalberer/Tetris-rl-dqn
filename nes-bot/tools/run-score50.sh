#!/bin/sh
# Play one original-game attempt with the 50M score bot (CEM tetris weights with a 3rd-piece expectation, no spawn wait, tetris mode
# at the 1G levels outside the crash zones) in MesenCE,
# then fresh-replay verify it.  Usage:
#   sh nes-bot/tools/run-score50.sh SEED [TARGET] [PLANNER_MASK EMULATOR_MASK]
#     TARGET: 255 (default), 511 (rebirth, then level 255 again) or "endless" (until top-out or crash)
# Paths: set MESEN (portable MesenCE 2.2.1 Mesen.exe), PY (emulator venv python with nes-py/torch/
# gym-tetris) and JIT_DEPS (folder with numba) or edit the defaults below. Output: runs/run-score50/seed<SEED>.
# For a frozen experiment, copy nes-bot/ to a separate folder first and run from there.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)  # repository root
NES="$ROOT/nes-bot"
SEED=${1:?seed (menu wait frames)}
TARGET=${2:-255}
PLANNER=${3:-1}
EMULATOR=${4:-2}
MESEN=${MESEN:-"$ROOT/../nes-late-fidelity/mesence-2.2.1/Mesen.exe"}
PY=${PY:-"$ROOT/../nes-original-env/.venv/Scripts/python.exe"}
JIT_DEPS=${JIT_DEPS:-"$ROOT/../nes-lookahead-deps"}
W="$NES/nes_original/weights"
if [ "$TARGET" = endless ]; then GOAL="--endless --max-frames 100000000 --process-timeout 1209600"
else GOAL="--target-level $TARGET --max-frames 4000000 --process-timeout 43200"; fi
OUT="$ROOT/runs/run-score50"; mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$NES"
exec "$PY" -m nes_original.endurance --output "$OUT/seed$SEED" --mesen "$MESEN" --jit-deps "$JIT_DEPS"   --selector tuned-crash-map --weights dellacherie --beam 5 --reaction-frames 0 --late-plan keep --alt-weights "$W/tetris-cem-b7-expect2.json" --seeds "$SEED" $GOAL   --zone-weights "$W/nosingles-156-158.json" --zone-levels 156 158 --zone-ban 2 --zone-expect 4   --extra-zone "$W/lowstack-246-249.json" 246 249 0 4 --extra-zone "$W/dellacherie.json" 250 254 0 4   --extra-zone "$W/dellacherie.json" 168 172 28 2 --extra-zone "$W/dellacherie.json" 173 173 4 2   --cpu-mask "$PLANNER" --emulator-cpu-mask "$EMULATOR" --transport pipe-blocking --verification-transport pipe   --nonactive-input neutral --stall-rescue-input a --stall-rescue-after 120 --stall-rescue-fallback right   --stall-rescue-primary-frames 8 --stall-grace-frames 2400   --no-softdrop-levels 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 250 251 252 254
