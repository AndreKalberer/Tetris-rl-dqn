"""Replay CLI option routing with mocked cores; no emulator or native actions."""
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import ANY,patch

from nes_original import replay


class ReplayCliTests(unittest.TestCase):
    @contextmanager
    def invocation(self,options=(),*,mesen=True):
        record = dict(protocol='original-ntsc-a-mesence-input-replay-v1' if mesen else
            'original-ntsc-a-input-replay-v1',input_hex='00018000',
            result=dict(level=167,lines=1610,placements=4031,over=False))
        path = Path.cwd()/'mock-recovery.json'
        with patch('sys.argv',['nes_original.replay',str(path),*options]),\
                patch.object(Path,'read_text',return_value=json.dumps(record)) as read,\
                patch('nes_original.replay.OriginalGame') as original,\
                patch('nes_original.mesen.MesenGame') as reference,\
                patch('nes_original.replay.subprocess.Popen') as popen,\
                redirect_stdout(io.StringIO()) as out,redirect_stderr(io.StringIO()) as err:
            yield dict(record=record,path=path,read=read,original=original,reference=reference,
                popen=popen,out=out,err=err)

    def assert_only_replay(self,context,factory):
        game = factory.return_value.__enter__.return_value
        game.load_replay.assert_called_once_with(context['record'],callback=ANY)
        self.assertTrue(callable(game.load_replay.call_args.kwargs['callback']))
        game.step.assert_not_called()
        game.reset.assert_not_called()
        game.screen.assert_not_called()
        context['popen'].assert_not_called()
        self.assertTrue(json.loads(context['out'].getvalue())['exact_ram_rgb_replay'])

    def test_mesen_defaults_remain_polling_pipe_and600_frame_host_watchdog(self):
        with self.invocation(['--mesen','Mesen.exe']) as c:
            replay.main()
            c['reference'].assert_called_once_with(executable=Path('Mesen.exe'),
                session_root=c['path'].parent/'reference-replay-sessions',transport='pipe',watchdog_frames=600)
            c['original'].assert_not_called()
            self.assert_only_replay(c,c['reference'])

    def test_recovery_threshold_and_each_explicit_transport_are_passed_without_extra_inputs(self):
        for transport in ('pipe','pipe-blocking','file'):
            with self.subTest(transport=transport),self.invocation(
                    ['--mesen','Mesen.exe','--watchdog-frames','3600','--transport',transport]) as c:
                before = deepcopy(c['record'])
                replay.main()
                c['reference'].assert_called_once_with(executable=Path('Mesen.exe'),
                    session_root=c['path'].parent/'reference-replay-sessions',
                    transport=transport,watchdog_frames=3600)
                self.assertEqual(c['record'],before)
                self.assert_only_replay(c,c['reference'])

    def test_original_backend_default_constructor_and_replay_are_unchanged(self):
        with self.invocation(mesen=False) as c:
            replay.main()
            c['original'].assert_called_once_with()
            c['reference'].assert_not_called()
            self.assert_only_replay(c,c['original'])

    def test_original_backend_rejects_explicit_new_mesen_only_flags(self):
        for options in (['--watchdog-frames','600'],['--transport','pipe'],
                        ['--mesen','Mesen.exe','--watchdog-frames','3600']):
            with self.subTest(options=options),self.invocation(options,mesen=False) as c:
                with self.assertRaises(SystemExit) as error: replay.main()
                self.assertEqual(error.exception.code,2)
                c['original'].assert_not_called()
                c['reference'].assert_not_called()
                c['popen'].assert_not_called()

    def test_minimum60_mirrors_adapter_and_smaller_or_noninteger_thresholds_fail(self):
        with self.invocation(['--mesen','Mesen.exe','--watchdog-frames','60']) as c:
            replay.main()
            self.assertEqual(c['reference'].call_args.kwargs['watchdog_frames'],60)
        for value in ('59','0','-1','3.5','True'):
            with self.subTest(value=value),self.invocation(['--mesen','Mesen.exe','--watchdog-frames',value]) as c:
                with self.assertRaises(SystemExit) as error: replay.main()
                self.assertEqual(error.exception.code,2)
                c['original'].assert_not_called()
                c['reference'].assert_not_called()
                c['read'].assert_not_called()

    def test_invalid_transport_and_missing_core_are_rejected_before_core_start(self):
        for options in (['--mesen','Mesen.exe','--transport','unknown'],
                        ['--watchdog-frames','3600','--transport','pipe-blocking']):
            with self.subTest(options=options),self.invocation(options) as c:
                with self.assertRaises(SystemExit) as error: replay.main()
                self.assertEqual(error.exception.code,2)
                c['original'].assert_not_called()
                c['reference'].assert_not_called()

    def test_replay_failure_propagates_and_is_not_printed_as_verified(self):
        with self.invocation(['--mesen','Mesen.exe','--watchdog-frames','3600']) as c:
            game = c['reference'].return_value.__enter__.return_value
            game.load_replay.side_effect = RuntimeError('record diverged')
            with self.assertRaisesRegex(RuntimeError,'record diverged'): replay.main()
            game.load_replay.assert_called_once_with(c['record'],callback=ANY)
            game.step.assert_not_called()
            self.assertEqual(c['out'].getvalue(),'')


if __name__=='__main__': unittest.main()
