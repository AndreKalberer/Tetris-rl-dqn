from dataclasses import replace
import numpy as np

from offline.engine import Game,Rules,Pose,cells,collides,rotate,advance
from offline.bot import enumerate_paths


def test_bags_and_seed_zero_are_deterministic():
    a,b=Game(0),Game(0)
    assert a.piece==b.piece and list(a.queue)==list(b.queue)
    # The first active piece plus buffered draws form complete seven-piece bags.
    seq=[a.piece]
    seq.extend(a.take_next() for _ in range(13))
    assert set(seq[:7])==set(range(7)) and set(seq[7:14])==set(range(7))


def test_gravity_is_driven_by_frames_without_input():
    game=Game(0)
    y=game.pose.y
    for _ in range(29): game.tick()
    assert game.pose.y==y
    game.tick()
    assert game.pose.y==y+1


def test_wall_collision_and_srs_kick_cases():
    board=np.zeros((24,10),dtype=np.uint8)
    t=Pose(-1,10,1)
    assert not collides(board,2,t)
    rotated=rotate(board,2,t,1)
    assert rotated.rotation==2 and rotated.x==0
    i=Pose(-2,10,1)
    assert not collides(board,0,i)
    assert rotate(board,0,i,-1).x==0
    floor=rotate(board,2,Pose(3,22),1)
    assert floor.rotation==1 and floor.y==21 and floor.x==2


def test_lock_delay_and_reset_limit():
    rules=Rules(gravity_frames=1000,lock_frames=3,max_lock_resets=2)
    game=Game(1,rules)
    game.piece=1
    game.pose=Pose(4,22)
    for action in ['none','left','none','right','none']:
        assert not game.tick(action)['locked']
    assert game.pose.resets==2
    assert game.tick('left')['locked']
    assert game.placements==1


def test_hold_once_per_piece_and_hard_drop_resets_it():
    game=Game(10)
    current=game.piece
    next_piece=game.queue[0]
    assert game.tick('hold')['accepted']
    assert game.held==current and game.piece==next_piece
    assert not game.tick('hold')['accepted']
    assert game.tick('hard')['locked']
    assert not game.hold_used


def test_line_clear_and_sprint_finish():
    game=Game(0,Rules(target_lines=4))
    game.piece=0
    game.pose=Pose(2,2,1)
    game.board[-4:]=2
    game.board[-4:,4]=0
    event=game.tick('hard')
    assert event['lines']==4 and game.lines==4 and game.won
    assert not game.visible.any()


def test_endurance_has_no_line_win_condition():
    game=Game(0,Rules(target_lines=None))
    game.piece=0
    game.pose=Pose(2,2,1)
    game.board[-4:]=2
    game.board[-4:,4]=0
    game.lines=100000
    assert game.tick('hard')['lines']==4
    assert game.lines==100004 and not game.won and not game.over


def test_spawn_collision_tops_out():
    game=Game(0)
    pose=game.pose
    dx,dy=cells(game.piece,0)[0]
    game.board[pose.y+dy,pose.x+dx]=1
    game.spawn(game.piece)
    assert game.over


def test_every_planned_path_replays_through_real_timed_inputs():
    for seed in [0,1,2]:
        base=Game(seed)
        predicted=base.pose
        for _ in range(base.rules.reaction_frames):
            predicted,locked,_=advance(base.board,base.piece,predicted,'none',base.rules)
            assert not locked
        planned=enumerate_paths(base,predicted)
        assert len(planned.paths)>10
        for i,path in enumerate(planned.paths):
            game=Game(seed)
            for _ in range(game.rules.reaction_frames): game.tick()
            for command in path:
                event=game.tick(command)
                assert event['accepted'],(seed,path,command)
                if event['locked']:
                    break
                for _ in range(game.rules.control_frames-1):
                    event=game.tick()
                    assert not event['locked']
            assert game.placements==1
            assert np.array_equal(game.board,planned.full_boards[i])
            assert game.lines==planned.candidates.lines[i]
