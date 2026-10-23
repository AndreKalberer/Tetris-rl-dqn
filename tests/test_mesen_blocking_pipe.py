"""Opt-in Windows pipe safety and native same-input transport comparison.

No ROM/bridge edits, policy choices, training, or live-worker changes.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest

from nes_original.mesen import MesenFault,MesenGame,_WindowsPipes


CLIENT = r'''
import ctypes as c,sys,time
k=c.WinDLL('kernel32',use_last_error=True)
k.CreateFileW.argtypes=[c.c_wchar_p,c.c_uint32,c.c_uint32,c.c_void_p,c.c_uint32,c.c_uint32,c.c_void_p]
k.CreateFileW.restype=c.c_void_p
k.WriteFile.argtypes=[c.c_void_p,c.c_void_p,c.c_uint32,c.POINTER(c.c_uint32),c.c_void_p]
k.ReadFile.argtypes=k.WriteFile.argtypes
k.CloseHandle.argtypes=[c.c_void_p]
handles=[]
try:
 for path,access in ((sys.argv[1],0x80000000),(sys.argv[2],0x40000000)):
  handle=k.CreateFileW(path,access,0,None,3,0,None)
  if handle==c.c_void_p(-1).value: raise c.WinError(c.get_last_error())
  handles.append(handle)
 command=c.create_string_buffer(1); received=c.c_uint32()
 if not k.ReadFile(handles[0],command,1,c.byref(received),None): raise c.WinError(c.get_last_error())
 if sys.argv[3]=='one-then-hung':
  text=b'{"sequence":1}\n'; count=c.c_uint32()
  if not k.WriteFile(handles[1],text,len(text),c.byref(count),None): raise c.WinError(c.get_last_error())
  time.sleep(60)
 elif sys.argv[3]=='fragments':
  for text in (b'{"sequence":',b'1}\n'):
   count=c.c_uint32()
   if not k.WriteFile(handles[1],text,len(text),c.byref(count),None): raise c.WinError(c.get_last_error())
   time.sleep(.03)
  time.sleep(.2)
 elif sys.argv[3]=='closed': time.sleep(.2)
 else: time.sleep(60)
finally:
 for handle in handles:k.CloseHandle(handle)
'''


@unittest.skipUnless(os.name=='nt','Windows native pipe APIs required')
class BlockingPipeSafetyTests(unittest.TestCase):
    def connected(self,mode):
        pipes = _WindowsPipes(blocking=True)
        self.addCleanup(pipes.close)
        child = subprocess.Popen([sys.executable,'-c',CLIENT,pipes.to_lua,pipes.from_lua,mode],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        def stop():
            if child.poll() is None: child.kill()
            child.wait(timeout=2)
        self.addCleanup(stop)
        pipes.connect(child,5)  # Includes cold Windows CPython process startup.
        pipes.send('go')  # The peer cannot exit before bounded server connect completes.
        return pipes,child

    def test_unconnected_startup_remains_bounded_before_switch_to_wait(self):
        pipes = _WindowsPipes(blocking=True)
        try:
            start = time.monotonic()
            with self.assertRaisesRegex(MesenFault,'connection timeout'):
                pipes.connect(SimpleNamespace(poll=lambda:None),.1)
            self.assertLess(time.monotonic()-start,1)
        finally: pipes.close()

    def test_blocking_reader_reassembles_fragmented_newline_message(self):
        pipes,child = self.connected('fragments')
        self.assertEqual(pipes.receive(),{'sequence':1})
        child.wait(timeout=2)

    def test_closed_client_unblocks_read_and_leaves_no_owned_process(self):
        pipes,child = self.connected('closed')
        start = time.monotonic()
        with self.assertRaises(OSError): pipes.receive()
        child.wait(timeout=2)
        self.assertLess(time.monotonic()-start,1.5)
        self.assertIsNotNone(child.poll())

    def test_hung_client_watchdog_kill_and_read_cancel_are_bounded(self):
        pipes,child = self.connected('hung')
        def watchdog():
            child.kill()
            pipes.abort_read()
        timer = threading.Timer(.1,watchdog)
        timer.start()
        try:
            start = time.monotonic()
            with self.assertRaises(OSError): pipes.receive()
            self.assertLess(time.monotonic()-start,1.5)
            child.wait(timeout=2)
        finally: timer.join(timeout=2)
        self.assertFalse(timer.is_alive())
        self.assertIsNotNone(child.poll())

    def test_concurrent_close_cancels_only_registered_read_before_freeing_handles(self):
        pipes,child = self.connected('hung')
        outcome = []
        def reader():
            try: pipes.receive()
            except OSError: outcome.append('cancelled')
        thread = threading.Thread(target=reader,daemon=True)
        thread.start()
        deadline = time.monotonic()+1
        while not pipes._read_pending and time.monotonic()<deadline: time.sleep(.001)
        start = time.monotonic()
        pipes.close()
        thread.join(timeout=1)
        self.assertLess(time.monotonic()-start,2.5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome,['cancelled'])
        self.assertFalse(pipes.handles)
        self.assertIsNone(pipes._read_thread)
        self.assertIsNone(pipes.cancel_error)

    def test_abort_before_native_read_entry_retries_without_canceling_later_caller_io(self):
        pipes,child = self.connected('hung')
        entered,release,finished = threading.Event(),threading.Event(),threading.Event()
        native_read,native_cancel = pipes.k.ReadFile,pipes.k.CancelSynchronousIo
        calls,outcome = [],[]
        def paused_read(*args):
            entered.set()
            if not release.wait(1): raise RuntimeError('Test pause deadline exceeded')
            return native_read(*args)
        def cancel_registered(handle):
            self.assertEqual(handle,pipes._read_thread)
            calls.append(time.monotonic())
            return native_cancel(handle)
        pipes.k.ReadFile,pipes.k.CancelSynchronousIo = paused_read,cancel_registered
        def reader():
            try: pipes.receive()
            except OSError: outcome.append('cancelled')
            finally: finished.set()
        reader_thread = threading.Thread(target=reader,daemon=True)
        reader_thread.start()
        self.assertTrue(entered.wait(1))
        abort_thread = threading.Thread(target=pipes.abort_read,daemon=True)
        abort_thread.start()
        time.sleep(.04)  # Cancel occurs while the registered thread has no native IO yet.
        self.assertTrue(calls)
        release.set()
        abort_thread.join(timeout=2.5)
        reader_thread.join(timeout=1)
        self.assertFalse(abort_thread.is_alive())
        self.assertFalse(reader_thread.is_alive())
        self.assertTrue(finished.is_set())
        self.assertEqual(outcome,['cancelled'])
        self.assertGreaterEqual(len(calls),2)
        count = len(calls)
        time.sleep(.03)
        self.assertEqual(len(calls),count)  # No cancellation continues after reader finally exits.
        self.assertIsNone(child.poll())  # Cancellation did not kill the client or another process.

    def test_recycled_native_thread_id_refreshes_owner_handle_before_cancellation(self):
        pipes,child = self.connected('one-then-hung')
        messages,outcome = [],[]
        first = threading.Thread(target=lambda:messages.append(pipes.receive()))
        first.start()
        first.join(timeout=1)
        self.assertFalse(first.is_alive())
        self.assertEqual(messages,[{'sequence':1}])
        self.assertIs(pipes._read_thread_owner,first)
        def second_reader():
            # Simulate Windows recycling the dead first thread's native ID.
            # The stored HANDLE still refers to that dead first thread.
            pipes._read_thread_id = threading.get_native_id()
            try: pipes.receive()
            except OSError: outcome.append('cancelled')
        second = threading.Thread(target=second_reader,daemon=True)
        second.start()
        deadline = time.monotonic()+1
        while not pipes._read_pending and time.monotonic()<deadline: time.sleep(.001)
        self.assertIs(pipes._read_thread_owner,second)
        pipes.abort_read()
        second.join(timeout=1)
        self.assertFalse(second.is_alive())
        self.assertEqual(outcome,['cancelled'])
        self.assertIsNone(child.poll())


@unittest.skipUnless(os.name=='nt' and os.environ.get('NES_MESEN_EXE') and
                    os.environ.get('NES_MESEN_BENCH_RECORD'),
                    'Opt in with existing core and a legal recorded input trace')
class BlockingPipeNativeTests(unittest.TestCase):
    def test_fixed480_frames_have_identical_pipe_and_blocking_native_proofs(self):
        source = json.loads(Path(os.environ['NES_MESEN_BENCH_RECORD']).read_text())
        inputs = bytes.fromhex(source['input_hex'])
        destination = Path(os.environ['NES_PIPE_BLOCKING_TEST_ROOT'])
        records,benchmark = {},{}
        for transport in ('pipe','pipe-blocking'):
            with MesenGame(executable=os.environ['NES_MESEN_EXE'],transport=transport,
                    session_root=destination/transport,affinity=1<<15,process_timeout=120) as game:
                game.reset(start_level=source['start_level'],seed=source['menu_wait_seed'])
                boot = len(game.inputs)
                self.assertEqual(bytes(game.inputs),inputs[:boot])
                actions = inputs[boot:game.start_frame+480]
                self.assertEqual(len(actions),479)
                start,cpu = time.perf_counter(),time.process_time()
                for action in actions: game.step(action)
                elapsed = time.perf_counter()-start
                benchmark[transport] = dict(native_step_calls=len(actions),
                    gameplay_frames=480,wall_seconds=elapsed,
                    host_process_cpu_seconds=time.process_time()-cpu,
                    native_steps_per_wall_second=len(actions)/elapsed)
                records[transport] = game.replay()
                self.assertEqual(records[transport]['result']['frames'],480)
        self.assertEqual(records['pipe'],records['pipe-blocking'])
        with MesenGame(executable=os.environ['NES_MESEN_EXE'],transport='pipe-blocking',
                session_root=destination/'fresh-replay',affinity=1<<15,process_timeout=120) as other:
            other.load_replay(records['pipe'])
            self.assertEqual(other.replay(),records['pipe'])
        result = dict(scope='One paired480-frame transport microbenchmark, not playing strength',
            full_input_ram_rgb_clock_records_identical=True,fresh_process_exact_replay=True,
            measurements=benchmark,speedup=benchmark['pipe']['wall_seconds']/
                benchmark['pipe-blocking']['wall_seconds'])
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        (destination/'pipe-replay.json').write_text(json.dumps(records['pipe'],indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)


if __name__=='__main__': unittest.main()
