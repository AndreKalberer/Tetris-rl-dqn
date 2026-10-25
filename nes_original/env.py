"""Read-only game instrumentation and genuine controller-driven startup.

No gym-tetris environment helpers, RAM writes, ROM patches, skipped gameplay
frames, save-state RNG reseeding, hold, or hard drop. nes-py runs every frame.
Native snapshots are in-memory only; portable saves reconstruct from inputs.
"""
from dataclasses import dataclass
from enum import IntFlag
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


ROM_SHA256 = 'acab10f2dbad4d4ef5d56b31cd4cea208bb6e2e2c5dba5eb339d711ec04e3535'
PROTOCOL = 'original-ntsc-a-input-replay-v1'


class Buttons(IntFlag):
    """nes-py controller bits (reverse of the game's internal bit order)."""
    A = 1
    B = 2
    SELECT = 4
    START = 8
    UP = 16
    DOWN = 32
    LEFT = 64
    RIGHT = 128


GAMEPLAY_MASK = int(Buttons.A | Buttons.B | Buttons.DOWN | Buttons.LEFT | Buttons.RIGHT)


def from_game_bits(buttons):
    """Convert the provisional controller's internal-format gameplay bits."""
    buttons = int(buttons)
    if buttons < 0 or buttons & ~0xC7:
        raise ValueError('Only Left/Right/Down/A/B are gameplay actions')
    return int(f'{buttons:08b}'[::-1], 2)


class ObservationFault(RuntimeError, ValueError):
    """Public RAM observation could not be decoded (e.g. corrupted BCD counters).

    RuntimeError routes it through evaluate's durable (AssertionError,
    RuntimeError) controller-error path; ValueError keeps existing callers and
    tests that expect ValueError unchanged. This is a decode failure, not a
    hardware crash diagnosis.
    """


def bcd(data):
    """Little-endian packed decimal, rejecting invalid decimal nibbles."""
    result = 0
    for index, byte in enumerate(data):
        byte = int(byte)
        if byte & 15 > 9 or byte >> 4 > 9:
            raise ObservationFault(f'Invalid packed decimal byte {byte:02x}')
        result += ((byte >> 4) * 10 + (byte & 15)) * 100**index
    return result


@dataclass(frozen=True)
class Observation:
    """Agent data only: board, active/preview pieces, controls, visible counters.

    This intentionally excludes the ROM RNG, CPU state, future pieces and RAM.
    Control timers are exposed for the declared RAM-state research interface.
    """
    board: np.ndarray
    x: int
    y: int
    orientation: int
    next_orientation: int
    phase: int
    level: int
    lines: int
    score: int
    placements: int
    frames: int
    start_level: int
    fall: int
    das: int
    soft: int
    soft_points: int
    previous: int
    pending: int
    over: bool


@dataclass(frozen=True)
class Snapshot:
    native: object
    inputs: bytes
    placements: int
    start_frame: int
    start_level: int
    seed: int
    started: bool


def bundled_rom():
    # Use the installed dependency asset in place; never copy it into this repo.
    import gym_tetris
    return Path(gym_tetris.__file__).parent / '_roms' / 'Tetris.nes'


class OriginalGame:
    """One input byte and one native emulator frame per step.

    `seed` means a reproducible number of idle frames on the level menu, not a
    synthetic RNG seed. Only $C0/$68/$67 are read to drive the genuine menus.
    ROM identity is enforced; emulator fidelity to physical hardware is separate.
    """
    def __init__(self, rom_path=None):
        self.rom_path = Path(rom_path) if rom_path else bundled_rom()
        self.rom_sha256 = hashlib.sha256(self.rom_path.read_bytes()).hexdigest()
        if self.rom_sha256 != ROM_SHA256:
            raise ValueError('Expected original Nintendo USA/NTSC Tetris image')
        self._env = None
        self.inputs = bytearray()
        self.started = False
        self.start_frame = self.placements = self.start_level = self.seed = 0

    def _step_raw(self, buttons):
        before = int(self._env.ram[0x68])
        self._env.step(int(buttons))
        self.inputs.append(int(buttons))
        if self.started and before == 2 and int(self._env.ram[0x68]) == 3:
            self.placements += 1

    def reset(self, *, start_level=18, seed=0):
        if not isinstance(start_level, int) or not 0 <= start_level <= 19:
            raise ValueError('Menu start level must be an integer from 0 to 19')
        if not isinstance(seed, int) or not 0 <= seed <= 32766:
            raise ValueError('Menu-wait seed must be an integer from 0 to 32766')
        # Native reset is a soft reset: prior RAM/PPU history can survive it.
        # Recreate the emulator for deterministic power-on, without RAM writes.
        from nes_py import NESEnv
        if self._env is not None:
            self._env.close()
        self._env = NESEnv(str(self.rom_path), render_mode='rgb_array')
        self._env.ram.flags.writeable = False
        self._env.reset(seed=0)  # Base-class reset; no subclass RAM hacks.
        self.inputs = bytearray()
        self.started = False
        self.placements = self.start_frame = 0
        self.start_level, self.seed = start_level, seed

        def wait(frames):
            for _ in range(frames):
                self._step_raw(0)

        def pulse(buttons):
            self._step_raw(buttons)
            wait(3)

        wait(120)
        for _ in range(1000):
            if int(self._env.ram[0xC0]) == 3:
                break
            pulse(Buttons.START)
        else:
            raise RuntimeError('Original-game level menu not reached')
        wait(20)
        for _ in range(start_level % 10):
            pulse(Buttons.RIGHT)
        if int(self._env.ram[0x67]) != start_level % 10:
            raise RuntimeError('Level-menu controller navigation failed')
        wait(seed)
        # Original game requires newly-pressed Start while A is already held.
        if start_level >= 10:
            self._step_raw(Buttons.A)
            self._step_raw(Buttons.A)
            self._step_raw(Buttons.A | Buttons.START)
        else:
            self._step_raw(Buttons.START)
        wait(1)
        for _ in range(500):
            ram = self._env.ram
            if int(ram[0xC0]) == 4 and int(ram[0x68]) == 1:
                break
            wait(1)
        else:
            raise RuntimeError('Original-game active play not reached')
        if int(self._env.ram[0x64]) != start_level:
            raise RuntimeError('Original-game starting level does not match')
        self.started = True
        self.start_frame = len(self.inputs)
        # The first active flag is initialized before the first control update.
        # Record one compulsory idle frame to enter the steady input pipeline;
        # it remains part of episode frames and the portable replay.
        self._step_raw(0)
        return self.observe()

    @property
    def over(self):
        return self.started and (int(self._env.ram[0x68]) == 10 or int(self._env.ram[0xC0]) != 4)

    def step(self, buttons=0):
        buttons = int(buttons)
        if not self.started or self.over:
            raise RuntimeError('Reset before stepping, and do not step a finished game')
        if buttons < 0 or buttons & ~GAMEPLAY_MASK:
            raise ValueError('Gameplay supports only Left/Right/Down/A/B')
        if buttons & int(Buttons.LEFT | Buttons.RIGHT) == int(Buttons.LEFT | Buttons.RIGHT):
            raise ValueError('Opposite horizontal directions are excluded by this protocol')
        self._step_raw(buttons)
        return self.observe()

    def observe(self):
        if not self.started:
            raise RuntimeError('Reset before observing gameplay')
        r = self._env.ram
        board = (r[0x400:0x4C8].reshape(20, 10) != 0xEF).copy()
        board.flags.writeable = False
        return Observation(board=board, x=int(r[0x60]), y=int(r[0x61]),
            orientation=int(r[0x62]), next_orientation=int(r[0xBF]), phase=int(r[0x68]),
            level=int(r[0x64]), lines=bcd(r[0x70:0x71]) + 100 * int(r[0x71]),
            score=bcd(r[0x73:0x76]), placements=self.placements,
            frames=len(self.inputs) - self.start_frame, start_level=self.start_level,
            fall=int(r[0x65]), das=int(r[0x66]), soft=int(r[0x6E]),
            soft_points=int(r[0x6F]), previous=int(r[0xB6]),
            pending=int(f'{self.inputs[-1]:08b}'[::-1], 2), over=self.over)

    def screen(self):
        return self._env.screen.copy()

    def fingerprint(self):
        """Diagnostics only; hidden RAM never enters the agent observation."""
        return dict(ram_sha256=hashlib.sha256(self._env.ram.tobytes()).hexdigest(),
            screen_sha256=hashlib.sha256(self._env.screen.tobytes()).hexdigest())

    def snapshot(self):
        return Snapshot(self._env.dump_state(), bytes(self.inputs), self.placements,
            self.start_frame, self.start_level, self.seed, self.started)

    def restore(self, state):
        self._env.load_state(state.native)
        self.inputs = bytearray(state.inputs)
        self.placements, self.start_frame = state.placements, state.start_frame
        self.start_level, self.seed, self.started = state.start_level, state.seed, state.started

    def replay(self):
        obs = self.observe()
        return dict(protocol=PROTOCOL, rom_sha256=self.rom_sha256,
            dependencies={name: version(name) for name in ('nes-py', 'gym-tetris', 'gymnasium', 'numpy')},
            start_level=self.start_level, menu_wait_seed=self.seed,
            start_frame=self.start_frame, input_hex=self.inputs.hex(),
            placements=self.placements, fingerprint=self.fingerprint(),
            result={key: getattr(obs, key) for key in
                ('level', 'lines', 'score', 'placements', 'frames', 'over')})

    def save_replay(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.replay(), indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)

    def load_replay(self, data, *, callback=None):
        """Portable exact continuation by boot+inputs; never pickle native pointers.

        Loading costs the elapsed episode length, unlike in-memory snapshots.
        Versions and whole-RAM/RGB hashes are checked before continuation.
        """
        if data['protocol'] != PROTOCOL or data['rom_sha256'] != self.rom_sha256:
            raise ValueError('Replay protocol or game identity differs')
        expected = {name: version(name) for name in data['dependencies']}
        if expected != data['dependencies']:
            raise ValueError('Replay dependencies differ; use the recorded environment')
        self.reset(start_level=data['start_level'], seed=data['menu_wait_seed'])
        inputs = bytes.fromhex(data['input_hex'])
        boot_length = len(self.inputs)
        if bytes(self.inputs) != inputs[:boot_length] or self.start_frame != data['start_frame']:
            raise ValueError('Boot controller trace did not reproduce')
        if callback:
            callback(self)
        for action in inputs[boot_length:]:
            self.step(action)
            if callback:
                callback(self)
        actual = self.replay()
        if (actual['fingerprint'] != data['fingerprint'] or actual['result'] != data['result']
                or self.placements != data['placements']):
            raise ValueError('Replay diverged from recorded game state')
        return self.observe()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
