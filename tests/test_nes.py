from dataclasses import replace
import json

import numpy as np
import pytest

from nes.engine import (Buttons,Control,Game,Phase,Randomizer,Rules,collides,
                        CELLS,control_frame,first_transition,gravity_frames,lfsr_step)


def blank():
    return np.zeros((20,10),dtype=np.uint8)


def test_nintendo_j_orientations_have_the_correct_chirality():
    assert set(CELLS[4])=={(0,-1),(0,0),(-1,1),(0,1)}
    assert set(CELLS[6])=={(0,-1),(1,-1),(0,0),(0,1)}


def test_lfsr_known_boot_vector_and_complete_period():
    state=0x8988
    for expected in (0x44C4,0x2262,0x1131,0x0898,0x044C,0x0226,0x0113,0x8089):
        state=lfsr_step(state)
        assert state==expected
    state=0x8988
    seen=set()
    while state not in seen:
        seen.add(state)
        state=lfsr_step(state)
    assert len(seen)==32767 and state==0x8988 and 0 not in seen


def test_piece_selection_uses_orientation_id_and_reroll_can_repeat():
    rng=Randomizer()
    rng.state,rng.count,rng.last=0x0700,0,2
    assert rng.draw()==14  # Repeated T rerolls to L, using previous orientation 2.
    assert rng.state==0x8380 and rng.count==1
    rng.state,rng.count,rng.last=0x0700,0,10
    assert rng.draw()==2  # Same first roll is accepted when previous piece was O.
    rng.state,rng.count,rng.last=0x0300,6,8
    assert rng.draw()==8  # Fallback is allowed to repeat the preceding Z.


def test_seed_zero_boot_pieces_and_frame_dependent_randomizer():
    game=Game(0)
    assert game.control.orientation==8 and game.next_orientation==18
    state=game._randomizer.state
    game.tick()
    assert game._randomizer.state==lfsr_step(state)
    assert Game(32767).get_state()['randomizer']==Game(0).get_state()['randomizer']


@pytest.mark.parametrize('level,frames',[(0,48),(1,43),(8,8),(9,6),(10,5),(12,5),
    (13,4),(16,3),(18,3),(19,2),(28,2),(29,1),(50,1)])
def test_ntsc_gravity_fixtures(level,frames):
    assert gravity_frames(level)==frames
    state=Control(orientation=10,soft=0)
    for _ in range(frames-1):
        state,failed=control_frame(blank(),state,0,level)
        assert state.y==0 and not failed
    state,failed=control_frame(blank(),state,0,level)
    assert state.y==1 and not failed


def test_startup_delay_is_96_frames_and_down_cancels_it():
    state=Control(orientation=10)
    for _ in range(96):
        state,_=control_frame(blank(),state,0,0)
        assert state.y==0
    state,_=control_frame(blank(),state,0,0)
    assert state.y==1
    state,_=control_frame(blank(),Control(orientation=10),Buttons.DOWN,0)
    assert state.soft==1
    for _ in range(2):
        state,_=control_frame(blank(),state,Buttons.DOWN,0)
    assert state.y==1


def test_das_16_then_6_and_blocked_wall_charge():
    state=Control(orientation=17,y=0,soft=0)
    xs=[]
    for _ in range(23):
        state,_=control_frame(blank(),state,Buttons.LEFT,0)
        xs.append(state.x)
    assert xs[:16]==[4]*16 and xs[16:22]==[3]*6 and xs[22]==2
    state=Control(x=0,orientation=17,soft=0)
    state,_=control_frame(blank(),state,Buttons.LEFT,0)
    assert state.x==0 and state.das==16
    state,_=control_frame(blank(),state,Buttons.LEFT,0)
    assert state.x==0 and state.das==16


def test_rotation_edges_a_priority_and_no_wall_or_floor_kicks():
    state=Control(orientation=2,y=5,soft=0)
    once,_=control_frame(blank(),state,Buttons.A|Buttons.B,0)
    twice,_=control_frame(blank(),once,Buttons.A,0)
    assert once.orientation==3 and twice.orientation==3
    wall=Control(x=0,y=5,orientation=17,soft=0)
    blocked,_=control_frame(blank(),wall,Buttons.A,0)
    assert blocked.orientation==17 and blocked.x==0
    floor=Control(x=5,y=19,orientation=18,soft=0)
    blocked,_=control_frame(blank(),floor,Buttons.A,0)
    assert blocked.orientation==18 and blocked.y==19


def test_down_blocks_horizontal_and_gravity_does_not_double_drop():
    state=Control(orientation=10,soft=2,fall=47)
    moved,_=control_frame(blank(),state,Buttons.DOWN,0)
    assert moved.y==1 and moved.soft_points==1
    moved,_=control_frame(blank(),Control(orientation=10,soft=0),Buttons.LEFT|Buttons.DOWN,0)
    assert moved.x==5


def test_lock_is_a_failed_drop_not_a_contact_timer():
    state=Control(x=5,y=18,orientation=10,soft=0)
    for _ in range(47):
        state,failed=control_frame(blank(),state,0,0)
        assert not failed
    state,failed=control_frame(blank(),state,0,0)
    assert failed
    _,failed=control_frame(blank(),replace(state,fall=0),0,29)
    assert failed


@pytest.mark.parametrize('y,delay',[(0,18),(4,18),(6,16),(10,14),(14,12),(18,10)])
def test_entry_delay_follows_render_and_scan_states(y,delay):
    game=Game()
    game.control=Control(x=5,y=y,orientation=10,fall=47,soft=0)
    if y<18:
        game.board[y+2,4:6]=1
    game.tick()
    assert game.phase==Phase.LOCK and game.placements==0
    for elapsed in range(1,25):
        event=game.tick()
        if event['spawned']:
            break
    assert elapsed==delay and game.placements==1 and game.lines==0


def test_clear_animation_alignment_and_four_row_scoring():
    delays=[]
    for phase in range(4):
        game=Game()
        game.frames=phase
        game.control=Control(x=5,y=18,orientation=10,fall=47,soft=0)
        game.board[19]=1
        game.board[19,4:6]=0
        game.tick()
        for elapsed in range(1,40):
            if game.tick()['spawned']:
                break
        assert game.lines==1 and game.score==40
        assert game.board[19,4:6].all() and not game.board[:19].any()
        delays.append(elapsed)
    assert sorted(delays)==[27,28,29,30]


@pytest.mark.parametrize('start,first',[(0,10),(5,60),(9,100),(10,100),(12,100),
                                     (15,100),(16,110),(18,130),(19,140)])
def test_level_transition_fixtures(start,first):
    assert first_transition(start)==first
    game=Game(0,Rules(start_level=start))
    game.phase=Phase.SCORE
    game.lines=first-1
    game.completed=1
    game.tick()
    assert game.lines==first and game.level==start+1
    assert game.score==40*(start+2)


def test_tetris_score_uses_transition_level_and_soft_drop_bcd_quirk():
    game=Game(0,Rules(start_level=18))
    game.phase=Phase.SCORE
    game.lines=129
    game.completed=4
    game.tick()
    assert game.lines==133 and game.level==19 and game.score==24000
    game=Game()
    game.phase=Phase.SCORE
    game.score_bytes=[9,0,0]
    game.control=replace(game.control,soft_points=10)
    game.tick()
    assert game.score==12  # Original BCD behavior: 9 + nine drop cells yields 12.


def test_hidden_active_cells_are_allowed_and_only_visible_cells_are_stored():
    assert not collides(blank(),5,0,17)
    assert collides(blank(),5,-1,17)
    game=Game()
    game.control=Control(orientation=17,soft=0)
    game.phase=Phase.LOCK
    game.tick()
    assert game.placements==1 and np.count_nonzero(game.board)==2


def test_game_over_is_checked_at_lock_state():
    game=Game()
    game.board[0]=1
    game.board[1]=1
    game.control=Control(orientation=10,fall=47,soft=0)
    game.tick()
    assert game.phase==Phase.LOCK and not game.over
    assert game.tick()['game_over'] and game.over and game.placements==0


def test_observation_hides_rng_and_state_resume_is_exact():
    game=Game(8)
    for i in range(1000):
        game.tick(int(Buttons.LEFT if i%80<30 else Buttons.DOWN))
    restored=Game.from_state(json.loads(json.dumps(game.get_state())))
    assert restored.signature()==game.signature()
    for i in range(200):
        buttons=Buttons.RIGHT if i%50<20 else Buttons.DOWN
        assert game.tick(buttons)==restored.tick(buttons)
        assert game.signature()==restored.signature()
    observation=game.observe()
    assert not hasattr(observation,'randomizer') and not hasattr(observation,'queue')
    with pytest.raises(ValueError):
        observation.board[0,0]=7
    with pytest.raises(ValueError):
        game.tick(8)  # Up/hard-drop is not a gameplay action.


def test_controller_probe_replays_all_state_and_respects_cutoff():
    from nes.evaluate import episode,replay
    row,game=episode(3,max_frames=2500,wall_seconds=10)
    assert row['placements']>5 and row['replay_verified']
    assert replay(row).signature()==game.signature()
    assert all(entry['buttons'] & 3!=3 for entry in row['inputs'])
    if row['alive_at_stop']:
        assert not row['completed'] and row['stopped_by']!='natural_top_out'
