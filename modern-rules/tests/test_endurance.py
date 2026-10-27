from offline import endurance


def test_progress_reporting_preserves_unicode_and_history(tmp_path,monkeypatch):
    root=tmp_path/'work/repo'
    (root/'modern-rules/offline').mkdir(parents=True)
    (tmp_path/'outputs').mkdir()
    output=root/'runs/endurance'
    output.mkdir(parents=True)
    progress=tmp_path/'outputs/progress.md'
    progress.write_text('Goal: Tetris — progress\n\nActive work: starting\n\nHistory: 20/20 ✓\n',encoding='utf-8')
    monkeypatch.setattr(endurance,'__file__',str(root/'modern-rules/offline/endurance.py'))
    status=dict(state='running',seed=1,placements=10,lines=4,wall_seconds=1,wall_budget_seconds=10)
    for _ in range(30):
        endurance.report(output,status)
    text=progress.read_text(encoding='utf-8')
    assert text.startswith('Goal: Tetris — progress')
    assert text.endswith('History: 20/20 ✓\n')
    assert len(text)<500
    assert (root/'progress.md').read_text(encoding='utf-8')==text


def test_replay_uses_relocated_adjacent_segments(tmp_path):
    from dataclasses import asdict,replace
    import json
    import numpy as np
    from offline.engine import Game,Rules
    from offline.evaluate import board_hash

    game=Game(7,replace(Rules(),target_lines=None))
    original=tmp_path/'original/first'
    original.mkdir(parents=True)
    (original/'protocol.json').write_text('invalid original path',encoding='utf-8')
    for name,parent in [('first',None),('second',str(original))]:
        folder=tmp_path/'moved'/name
        folder.mkdir(parents=True)
        event=game.tick('hard')
        protocol=dict(seed=7,rules=asdict(game.rules),parent_segment=parent)
        result=dict(frames=game.frames,lines=game.lines,placements=game.placements,
                    final_board_sha256=board_hash(game.board))
        (folder/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
        (folder/'results.json').write_text(json.dumps(result),encoding='utf-8')
        (folder/'inputs.jsonl').write_text(json.dumps(dict(frame=game.frames,action='hard',accepted=event['accepted']))+'\n',encoding='utf-8')
        (folder/'locks.jsonl').write_text(json.dumps(dict(frame=game.frames,lines=game.lines,board_sha256=board_hash(game.board)))+'\n',encoding='utf-8')
    replayed=endurance.replay_segment(tmp_path/'moved/second')
    assert replayed.placements==2 and replayed.frames==2
    assert np.array_equal(replayed.board,game.board)
