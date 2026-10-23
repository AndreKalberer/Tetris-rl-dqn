"""Bounded, read-only original-game reference driver for portable MesenCE.

Requires an existing Windows portable MesenCE executable with settings.json
beside it. No downloads/installs are performed. Each inputPolled callback stops
before submitting the next input; step supplies that input and waits for the
next callback. This has no nes-py-style pending-input frame (input_pipeline=0).
Named-pipe IPC is the default; transport='file' retains the tested fallback.
transport='pipe-blocking' opts into cancellable, watchdog-bounded blocking reads.
These are bounded reference interfaces, not a high-throughput training service.
CPU/PPU/hardware parity and full late-game crash behavior remain separate tests.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

import numpy as np

from nes_original.env import Buttons, GAMEPLAY_MASK, Observation, ROM_SHA256, bcd, bundled_rom

PROTOCOL = 'original-ntsc-a-mesence-input-replay-v1'
PROFILE = ('--nes.region=Ntsc', '--nes.rampoweronstate=AllZeros',
    '--nes.randomizecpuppualignment=false', '--nes.randomizemapperpoweronstate=false',
    '--nes.ppuextrascanlinesbeforenmi=0', '--nes.ppuextrascanlinesafternmi=0',
    '--nes.removespritelimit=false', '--nes.enablehdpacks=false', '--emulation.runaheadframes=0')


class MesenFault(RuntimeError):
    """Reference-process/IPC fault; never a normal game-over result."""


class SuspectedCrash(MesenFault):
    """Bounded watchdog classification, not a hardware crash diagnosis."""


@dataclass(frozen=True)
class MesenObservation(Observation):
    input_pipeline: int = 0


class _WindowsPipes:
    """Two byte-stream pipes, each with one writer and one reader.

    Default server reads use PeekNamedPipe polling. The opt-in blocking reader
    uses registered synchronous IO, repeated cancellation, and the independent
    owner-process watchdog; connection and command writes remain nonblocking.
    Lua uses ordinary blocking stdio in both modes.
    """
    def __init__(self, *, blocking=False):
        import ctypes as c
        self.c = c
        self.k = k = c.WinDLL('kernel32', use_last_error=True)
        k.CreateNamedPipeW.argtypes = [c.c_wchar_p,c.c_uint32,c.c_uint32,c.c_uint32,
            c.c_uint32,c.c_uint32,c.c_uint32,c.c_void_p]
        k.CreateNamedPipeW.restype = c.c_void_p
        k.ConnectNamedPipe.argtypes = [c.c_void_p,c.c_void_p]
        k.ConnectNamedPipe.restype = c.c_int
        k.PeekNamedPipe.argtypes = [c.c_void_p,c.c_void_p,c.c_uint32,c.c_void_p,
            c.POINTER(c.c_uint32),c.c_void_p]
        k.PeekNamedPipe.restype = c.c_int
        for name in ('ReadFile','WriteFile'):
            fn = getattr(k,name)
            fn.argtypes = [c.c_void_p,c.c_void_p,c.c_uint32,c.POINTER(c.c_uint32),c.c_void_p]
            fn.restype = c.c_int
        k.CloseHandle.argtypes = [c.c_void_p]
        k.CloseHandle.restype = c.c_int
        self.blocking = blocking
        self._read_lock = threading.Lock()
        self._read_done = threading.Event()
        self._read_done.set()
        self._read_pending = self._aborted = False
        self._read_thread = self._read_thread_id = None
        self._read_thread_owner = None
        self.cancel_error = None
        if blocking:
            k.SetNamedPipeHandleState.argtypes = [c.c_void_p,c.POINTER(c.c_uint32),c.c_void_p,c.c_void_p]
            k.SetNamedPipeHandleState.restype = c.c_int
            k.OpenThread.argtypes = [c.c_uint32,c.c_int,c.c_uint32]
            k.OpenThread.restype = c.c_void_p
            k.CancelSynchronousIo.argtypes = [c.c_void_p]
            k.CancelSynchronousIo.restype = c.c_int
        token = uuid.uuid4().hex
        self.to_lua = '\\\\.\\pipe\\nes-ref-' + token + '-in'
        self.from_lua = '\\\\.\\pipe\\nes-ref-' + token + '-out'
        self.handles = []
        self.buffer = b''
        try:
            for name,access in ((self.from_lua,1),(self.to_lua,2)):
                # Duplex access grants mode-change rights on the receiving
                # server handle; protocol IO remains one reader/one writer.
                if blocking and access==1: access = 3  # PIPE_ACCESS_DUPLEX
                handle = k.CreateNamedPipeW(name,access,1,1,65536,65536,0,None)
                if handle == c.c_void_p(-1).value: raise c.WinError(c.get_last_error())
                self.handles.append(handle)
        except Exception:
            self.close(); raise

    def connect(self, process, timeout):
        pending = set(self.handles)
        deadline = time.monotonic()+timeout
        while pending and time.monotonic()<deadline:
            for handle in tuple(pending):
                ok = self.k.ConnectNamedPipe(handle,None)
                error = self.c.get_last_error()
                if ok or error == 535: pending.remove(handle)  # ERROR_PIPE_CONNECTED
                elif error not in (536,232): raise self.c.WinError(error)
            if process.poll() is not None: raise MesenFault('Mesen exited while connecting IPC')
            if pending: time.sleep(.001)
        if pending: raise MesenFault('Named-pipe connection timeout')
        if self.blocking:
            mode = self.c.c_uint32(0)  # PIPE_READMODE_BYTE | PIPE_WAIT; writer stays NOWAIT.
            if not self.k.SetNamedPipeHandleState(self.handles[0],self.c.byref(mode),None,None):
                raise self.c.WinError(self.c.get_last_error())

    def send(self, text):
        data = (text+'\n').encode('ascii')
        count = self.c.c_uint32()
        if not self.k.WriteFile(self.handles[1],data,len(data),self.c.byref(count),None):
            raise self.c.WinError(self.c.get_last_error())
        if count.value != len(data): raise MesenFault('Partial controller command write')

    def receive(self):
        if self.blocking:
            return self._receive_blocking()
        if b'\n' not in self.buffer:
            available = self.c.c_uint32()
            if not self.k.PeekNamedPipe(self.handles[0],None,0,None,self.c.byref(available),None):
                raise self.c.WinError(self.c.get_last_error())
            if available.value:
                count = self.c.c_uint32()
                buf = self.c.create_string_buffer(available.value)
                if not self.k.ReadFile(self.handles[0],buf,len(buf),self.c.byref(count),None):
                    raise self.c.WinError(self.c.get_last_error())
                self.buffer += buf.raw[:count.value]
        if b'\n' in self.buffer:
            line,self.buffer = self.buffer.split(b'\n',1)
            return json.loads(line)
        return None

    def _receive_blocking(self):
        # ctypes.WinDLL releases the GIL during ReadFile, so the independent
        # Mesen watchdog can kill its owned child and cancel this exact thread.
        with self._read_lock:
            if self._aborted: raise self.c.WinError(995)
        while b'\n' not in self.buffer:
            with self._read_lock:
                if self._aborted: raise self.c.WinError(995)  # ERROR_OPERATION_ABORTED
                if self._read_pending: raise MesenFault('Concurrent pipe reads are unsupported')
                thread_id = threading.get_native_id()
                owner = threading.current_thread()
                if thread_id != self._read_thread_id or owner is not self._read_thread_owner:
                    if self._read_thread is not None: self.k.CloseHandle(self._read_thread)
                    self._read_thread = self.k.OpenThread(1,False,thread_id)  # THREAD_TERMINATE
                    if not self._read_thread: raise self.c.WinError(self.c.get_last_error())
                    self._read_thread_id = thread_id
                    self._read_thread_owner = owner
                handle = self.handles[0]
                self._read_pending = True
                self._read_done.clear()
            try:
                count = self.c.c_uint32()
                buf = self.c.create_string_buffer(65536)
                if not self.k.ReadFile(handle,buf,len(buf),self.c.byref(count),None):
                    raise self.c.WinError(self.c.get_last_error())
                if not count.value: raise self.c.WinError(109)  # ERROR_BROKEN_PIPE
                if self._aborted: raise self.c.WinError(995)
                self.buffer += buf.raw[:count.value]
                if len(self.buffer)>1024*1024:
                    raise MesenFault('Named-pipe response exceeds bounded message size')
            finally:
                with self._read_lock:
                    self._read_pending = False
                    self._read_done.set()
        line,self.buffer = self.buffer.split(b'\n',1)
        return json.loads(line)

    def abort_read(self):
        """Permanently abort this session's read; never cancel another thread's work."""
        if not self.blocking: return
        deadline = time.monotonic()+2
        while True:
            with self._read_lock:
                self._aborted = True
                handle = self._read_thread if self._read_pending else None
                # Hold the lock through cancellation: the reader cannot leave
                # finally and start unrelated IO on the registered thread.
                if handle is not None and not self.k.CancelSynchronousIo(handle):
                    error = self.c.get_last_error()
                    if error!=1168: self.cancel_error = error  # ERROR_NOT_FOUND is a completion race.
            if self._read_done.wait(.01): return
            # A one-shot cancel can race before native ReadFile entry. Repeat
            # while registered; owned-child termination also closes the peer.
            if time.monotonic()>=deadline:
                raise MesenFault('Blocking pipe read did not cancel; active handle retained')

    def close(self):
        if self.blocking:
            self.abort_read()
            if self._read_thread is not None:
                self.k.CloseHandle(self._read_thread)
                self._read_thread = self._read_thread_id = None
                self._read_thread_owner = None
        for handle in self.handles: self.k.CloseHandle(handle)
        self.handles = []


class MesenGame:
    """OriginalGame-compatible gameplay API, using only genuine controller input.

    seed is an idle-frame count on the real menu, not an injected RNG seed.
    Sessions retain journals/diagnostics in session_root; ROM bytes stay in place.
    Native binary snapshots are deliberately not part of this small driver.
    affinity defaults to logical CPUs 12/14 on the audited workstation; callers
    on other machines must pass an appropriate available mask (or None).
    """
    def __init__(self, rom_path=None, *, executable=None, session_root=None,
                 action_timeout=30, process_timeout=3600, watchdog_frames=600,
                 affinity=0x5000, transport='pipe'):
        if os.name != 'nt':
            raise RuntimeError('This reference launcher currently supports Windows only')
        self.executable = Path(executable or os.environ.get('NES_MESEN_EXE', '')).resolve()
        if not self.executable.is_file() or not self.executable.with_name('settings.json').is_file():
            raise ValueError('Pass an existing portable MesenCE executable with local settings.json')
        # Reproducibility cannot silently inherit a customized portable profile.
        settings = json.loads(self.executable.with_name('settings.json').read_text(encoding='utf-8-sig'))
        if settings != {}:
            raise ValueError('Use a dedicated portable profile with settings.json containing {}')
        self.rom_path = Path(rom_path) if rom_path else bundled_rom()
        self.rom_sha256 = hashlib.sha256(self.rom_path.read_bytes()).hexdigest()
        if self.rom_sha256 != ROM_SHA256:
            raise ValueError('Expected original Nintendo USA/NTSC Tetris image')
        self.bridge = Path(__file__).with_name('mesen_bridge.lua')
        dependencies = ('MesenCore.dll','MesenNesDB.txt')
        if not all(self.executable.with_name(name).is_file() for name in dependencies):
            raise ValueError('Initialize the dedicated portable build once before using the reference adapter')
        self.identity = dict(executable_sha256=hashlib.sha256(self.executable.read_bytes()).hexdigest(),
            core_sha256=hashlib.sha256(self.executable.with_name('MesenCore.dll').read_bytes()).hexdigest(),
            database_sha256=hashlib.sha256(self.executable.with_name('MesenNesDB.txt').read_bytes()).hexdigest(),
            bridge_sha256=hashlib.sha256(self.bridge.read_bytes()).hexdigest(), profile=list(PROFILE))
        self.session_root = Path(session_root or Path.cwd() / 'runs' / 'mesen-reference').resolve()
        if action_timeout < 2 or process_timeout <= action_timeout or watchdog_frames < 60:
            raise ValueError('Require action_timeout>=2, larger process_timeout and watchdog_frames>=60')
        self.action_timeout, self.process_timeout = int(action_timeout), int(process_timeout)
        self.watchdog_frames, self.affinity = int(watchdog_frames), affinity
        if transport not in ('file','pipe','pipe-blocking'):
            raise ValueError('Choose file, pipe or pipe-blocking IPC')
        self.transport = transport
        self._pipes = None
        self._idle_stop = None
        self._idle_thread = None
        self._idle_deadline = 0
        self._idle_expired = False
        self._process_expired = False
        self._process = None
        self._logs = []
        self.session_dir = None
        self.inputs = bytearray()
        self.started = False
        self.start_frame = self.placements = self.start_level = self.seed = 0
        self._request = None
        self._ram = None
        self._capture_id = 0
        self._nmi_stalled = self._semantic_stalled = 0

    def _publish(self, text):
        self._idle_deadline = time.monotonic()+self.action_timeout
        if self._pipes is not None:
            try: self._pipes.send(text)
            except OSError as error:
                raise MesenFault(f'Named-pipe command failed; idle_timeout={self._idle_expired}; session={self.session_dir}') from error
            return
        target = self.session_dir / 'response.txt'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(text, encoding='ascii')
        deadline = time.monotonic() + self.action_timeout
        while True:
            try:
                temporary.replace(target)
                return
            except PermissionError:
                # Lua's short-lived read handle excludes delete sharing on Windows.
                if time.monotonic() >= deadline: raise
                time.sleep(.001)

    def _wait_request(self, after=0):
        deadline = time.monotonic() + self.action_timeout + 3
        while time.monotonic() < deadline:
            error = self.session_dir / 'error.json'
            if error.is_file():
                raise MesenFault(f'Lua bridge fault: {error.read_text()}; session={self.session_dir}')
            request = self.session_dir / 'request.json'
            if self._pipes is not None:
                try: data = self._pipes.receive()
                except OSError as error:
                    raise MesenFault(f'Named-pipe read failed; idle_timeout={self._idle_expired}; session={self.session_dir}') from error
            elif request.is_file():
                try:
                    data = json.loads(request.read_text(encoding='ascii'))
                except (OSError, ValueError):
                    data = None
            else: data = None
            if data is not None and data['sequence'] > after:
                if data['sequence'] != after + 1:
                    raise MesenFault('Input sequence skipped; refusing an unjournaled frame')
                self._request = data
                self._ram = np.frombuffer(bytes.fromhex(data['ram_hex']), dtype=np.uint8)
                if self._ram.size != 2048:
                    raise MesenFault('Expected exactly 2 KiB NES internal RAM')
                self._idle_deadline = time.monotonic()+self.action_timeout
                return
            if self._process.poll() is not None:
                raise MesenFault(f'Mesen exited {self._process.returncode}; session={self.session_dir}')
            time.sleep(.001)
        raise MesenFault(f'Input callback timeout; session={self.session_dir}')

    def _launch(self):
        self.close()
        self.session_dir = self.session_root / uuid.uuid4().hex
        self.session_dir.mkdir(parents=True, exist_ok=False)
        entry = self.session_dir / 'entry.lua'
        pipe_config = ''
        if self.transport in ('pipe','pipe-blocking'):
            self._pipes = _WindowsPipes(blocking=self.transport=='pipe-blocking')
            pipe_config = 'MESEN_BRIDGE_PIPE_IN=' + json.dumps(self._pipes.to_lua) + '\n'
            pipe_config += 'MESEN_BRIDGE_PIPE_OUT=' + json.dumps(self._pipes.from_lua) + '\n'
        entry.write_text('MESEN_BRIDGE_ROOT=' + json.dumps(self.session_dir.as_posix() + '/') + '\n'
            + f'MESEN_BRIDGE_TIMEOUT={self.action_timeout}\n'
            + pipe_config
            + 'dofile(' + json.dumps(self.bridge.as_posix()) + ')\n', encoding='utf-8')
        command = [str(self.executable), '--testRunner', f'--timeout={self.process_timeout}',
            '--enableStdout', '--doNotSaveSettings', '--debug.scriptwindow.allowioosaccess=true',
            f'--debug.scriptwindow.scripttimeout={self.action_timeout+5}', *PROFILE,
            str(entry), str(self.rom_path.resolve())]
        (self.session_dir / 'launch.json').write_text(json.dumps(dict(command=command,
            identity=self.identity, rom_sha256=self.rom_sha256, affinity=self.affinity), indent=2), encoding='utf-8')
        self._logs = [(self.session_dir / name).open('wb') for name in ('stdout.log', 'stderr.log')]
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        self._process = subprocess.Popen(command, cwd=self.executable.parent,
            stdout=self._logs[0], stderr=self._logs[1], startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        if self._pipes is not None:
            # Lua's blocking C read cannot run its own timeout hook. This
            # independent host thread bounds idle waits and kills only our child.
            self._idle_deadline = time.monotonic()+self.action_timeout
            self._idle_expired = False
            self._process_expired = False
            self._idle_stop = threading.Event()
            process,stop,pipes = self._process,self._idle_stop,self._pipes
            process_deadline = time.monotonic()+self.process_timeout
            def idle_watchdog():
                while not stop.wait(.1):
                    if process.poll() is not None: return
                    if time.monotonic()>process_deadline:
                        self._process_expired = True
                        process.kill()
                        pipes.abort_read()
                        return
                    if time.monotonic()>self._idle_deadline:
                        self._idle_expired = True
                        process.kill()
                        pipes.abort_read()
                        return
            self._idle_thread = threading.Thread(target=idle_watchdog,daemon=True)
            self._idle_thread.start()
        if self.affinity is not None:
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel.SetProcessAffinityMask.restype = ctypes.c_int
            if not kernel.SetProcessAffinityMask(int(self._process._handle), self.affinity):
                self.close()
                raise OSError(ctypes.get_last_error(), 'Cannot apply Mesen process affinity')
        if self._pipes is not None:
            self._pipes.connect(self._process,self.action_timeout)
        self._wait_request()

    def _step_raw(self, buttons):
        before = self._ram
        sequence = self._request['sequence']
        self._publish(f'{sequence} action {int(buttons)}')
        self._wait_request(sequence)
        self.inputs.append(int(buttons))
        if self.started:
            if int(before[0x68]) == 2 and int(self._ram[0x68]) == 3:
                self.placements += 1
            self._watchdog(before)

    def _watchdog(self, before):
        r = self._ram
        # A halted CPU can keep the PPU/input callbacks moving while RAM freezes.
        previous_nmi = int(before[0xB1]) + 256 * int(before[0xB2])
        current_nmi = int(r[0xB1]) + 256 * int(r[0xB2])
        self._nmi_stalled = self._nmi_stalled + 1 if previous_nmi == current_nmi else 0
        same = np.array_equal(before[0x60:0x80], r[0x60:0x80]) and np.array_equal(before[0x400:0x4C8], r[0x400:0x4C8])
        self._semantic_stalled = self._semantic_stalled + 1 if same else 0
        if int(r[0x68]) == 10:
            return
        reason = None
        if int(r[0xC0]) != 4:
            reason = 'unexpected_exit_from_gameplay'
        elif self._nmi_stalled >= self.watchdog_frames:
            reason = 'rom_nmi_counter_stalled'
        elif self._semantic_stalled >= self.watchdog_frames:
            reason = 'gameplay_state_stalled'
        if reason:
            self._fault_record(reason)
            raise SuspectedCrash(f'{reason}; diagnostics={self.session_dir}')

    def _fault_record(self, reason):
        data = dict(reason=reason, classification='suspected_crash', frames=len(self.inputs)-self.start_frame,
            input_hex=self.inputs.hex(), request=self._request)
        (self.session_dir / 'suspected-crash.json').write_text(json.dumps(data, indent=2), encoding='utf-8')

    def reset(self, *, start_level=18, seed=0):
        if not isinstance(start_level, int) or not 0 <= start_level <= 19:
            raise ValueError('Menu start level must be an integer from 0 to 19')
        if not isinstance(seed, int) or not 0 <= seed <= 32766:
            raise ValueError('Menu-wait seed must be an integer from 0 to 32766')
        self.started = False
        self.inputs = bytearray()
        self.placements = self.start_frame = 0
        self._nmi_stalled = self._semantic_stalled = 0
        self.start_level, self.seed = start_level, seed
        self._launch()

        def wait(frames):
            for _ in range(frames): self._step_raw(0)

        def pulse(buttons):
            self._step_raw(buttons); wait(3)

        wait(120)
        for _ in range(1000):
            if int(self._ram[0xC0]) == 3: break
            pulse(Buttons.START)
        else: raise MesenFault('Original-game level menu not reached')
        wait(20)
        for _ in range(start_level % 10): pulse(Buttons.RIGHT)
        if int(self._ram[0x67]) != start_level % 10:
            raise MesenFault('Level-menu controller navigation failed')
        wait(seed)
        if start_level >= 10:
            self._step_raw(Buttons.A); self._step_raw(Buttons.A)
            self._step_raw(Buttons.A | Buttons.START)
        else: self._step_raw(Buttons.START)
        wait(1)
        for _ in range(500):
            if int(self._ram[0xC0]) == 4 and int(self._ram[0x68]) == 1: break
            wait(1)
        else: raise MesenFault('Original-game active play not reached')
        if int(self._ram[0x64]) != start_level:
            raise MesenFault('Original-game starting level does not match')
        self.started = True
        self.start_frame = len(self.inputs)
        self._step_raw(0)
        return self.observe()

    @property
    def over(self):
        return self.started and int(self._ram[0x68]) == 10

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
        if not self.started: raise RuntimeError('Reset before observing gameplay')
        r = self._ram
        board = (r[0x400:0x4C8].reshape(20, 10) != 0xEF).copy()
        board.flags.writeable = False
        return MesenObservation(board=board, x=int(r[0x60]), y=int(r[0x61]),
            orientation=int(r[0x62]), next_orientation=int(r[0xBF]), phase=int(r[0x68]),
            level=int(r[0x64]), lines=bcd(r[0x70:0x71])+100*int(r[0x71]),
            score=bcd(r[0x73:0x76]), placements=self.placements,
            frames=len(self.inputs)-self.start_frame, start_level=self.start_level,
            fall=int(r[0x65]), das=int(r[0x66]), soft=int(r[0x6E]),
            soft_points=int(r[0x6F]), previous=int(r[0xB6]), pending=0, over=self.over)

    def _capture(self):
        self._capture_id += 1
        command = self._capture_id
        sequence = self._request['sequence']
        self._publish(f'{sequence} capture {command}')
        deadline = time.monotonic() + self.action_timeout
        while time.monotonic() < deadline:
            ack = self.session_dir / 'capture.json'
            if ack.is_file():
                try: data = json.loads(ack.read_text(encoding='ascii'))
                except (OSError, ValueError): data = None
                if data and data['sequence'] == sequence and data['command'] == command:
                    if data['pixels'] != 256*240: raise MesenFault('Unexpected NES screen dimensions')
                    return self.session_dir / f'screen-{command}'
            if self._process.poll() is not None: raise MesenFault('Mesen exited during screen capture')
            time.sleep(.001)
        raise MesenFault('Screen capture timeout')

    def screen(self):
        """Explicit capture of current completed frame; no emulation advance."""
        path = self._capture().with_suffix('.argb')
        pixels = np.frombuffer(path.read_bytes(), dtype='<u4').reshape(240, 256)
        return np.stack(((pixels>>16)&255, (pixels>>8)&255, pixels&255), axis=-1).astype(np.uint8)

    def fingerprint(self):
        return dict(ram_sha256=hashlib.sha256(self._ram.tobytes()).hexdigest(),
            screen_sha256=hashlib.sha256(self.screen().tobytes()).hexdigest(),
            clock={k:self._request[k] for k in ('pc','cpu_cycle','video_frame','ppu_scanline','ppu_cycle')})

    def replay(self):
        obs = self.observe()
        return dict(protocol=PROTOCOL, rom_sha256=self.rom_sha256, identity=self.identity,
            start_level=self.start_level, menu_wait_seed=self.seed, start_frame=self.start_frame,
            input_hex=self.inputs.hex(), placements=self.placements, fingerprint=self.fingerprint(),
            result={k:getattr(obs,k) for k in ('level','lines','score','placements','frames','over')})

    def save_replay(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(self.replay(), indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)

    def load_replay(self, data, *, callback=None, expected_fault=None):
        """Reconstruct exact input/state evidence, optionally including a final fault.

        expected_fault is an explicit caller-supplied mapping with type
        'SuspectedCrash' and one existing watchdog reason. It accepts that exact
        fault only on the final recorded input, after full input/result/RAM/RGB/
        clock comparisons. This verifies a fault trace, never a gameplay goal.
        Without this argument watchdog exceptions retain their normal behavior.
        """
        if expected_fault is not None:
            reasons = {'rom_nmi_counter_stalled','gameplay_state_stalled','unexpected_exit_from_gameplay'}
            if (not isinstance(expected_fault,dict) or set(expected_fault)!={'type','reason'} or
                    expected_fault['type']!='SuspectedCrash' or
                    not isinstance(expected_fault['reason'],str) or expected_fault['reason'] not in reasons):
                raise ValueError('Expected an explicit SuspectedCrash type and recognized watchdog reason')
        if data['protocol'] != PROTOCOL or data['rom_sha256'] != self.rom_sha256 or data['identity'] != self.identity:
            raise ValueError('Replay game, core executable, bridge or settings profile differs')
        self.reset(start_level=data['start_level'], seed=data['menu_wait_seed'])
        inputs = bytes.fromhex(data['input_hex'])
        boot_length = len(self.inputs)
        if bytes(self.inputs) != inputs[:boot_length] or self.start_frame != data['start_frame']:
            raise ValueError('Boot controller trace did not reproduce')
        if callback: callback(self)
        fault_seen = False
        for index,buttons in enumerate(inputs[boot_length:],start=boot_length):
            try:
                self.step(buttons)
            except SuspectedCrash as exc:
                if expected_fault is None:
                    raise
                reason = str(exc).partition('; diagnostics=')[0]
                if (type(exc) is not SuspectedCrash or reason!=expected_fault['reason'] or
                        index!=len(inputs)-1 or len(self.inputs)!=len(inputs)):
                    raise ValueError('Replay fault type, reason or final input position differs') from exc
                fault_seen = True
            if callback: callback(self)
        if expected_fault is not None and not fault_seen:
            raise ValueError('Expected replay fault did not occur on the final recorded input')
        actual = self.replay()
        if actual['fingerprint'] != data['fingerprint'] or actual['result'] != data['result']:
            raise ValueError('Replay diverged from recorded RAM, RGB, clock or game result')
        if expected_fault is not None and (bytes(self.inputs)!=inputs or
                actual['input_hex']!=data['input_hex'] or self.placements!=data['placements']):
            raise ValueError('Fault replay input journal or placement count differs')
        return self.observe()

    def snapshot(self):
        raise NotImplementedError('Native snapshots deferred; use an input replay for exact reconstruction')

    def restore(self, snapshot):
        raise NotImplementedError('Native snapshots deferred; use load_replay')

    def close(self):
        if self._idle_stop is not None:
            self._idle_stop.set()
        if self._idle_thread is not None:
            self._idle_thread.join(timeout=3 if self.transport=='pipe-blocking' else 1)
            if self.transport=='pipe-blocking' and self._idle_thread.is_alive():
                raise MesenFault('Blocking pipe watchdog did not exit within its cancellation bound')
            self._idle_thread = None
        self._idle_stop = None
        if self._process is not None:
            if self._process.poll() is None:
                try:
                    self._publish('quit')
                    self._process.wait(timeout=3)
                except (MesenFault, OSError, subprocess.TimeoutExpired):
                    self._process.kill(); self._process.wait(timeout=3)
            self._process = None
        if self._pipes is not None:
            self._pipes.close()
            self._pipes = None
        for log in self._logs: log.close()
        self._logs = []

    def __enter__(self): return self

    def __exit__(self, *args): self.close()
