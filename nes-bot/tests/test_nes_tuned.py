"""Fast simulator / two-mode evaluator checks (needs the optional numba deps)."""
from types import SimpleNamespace
import unittest

import numpy as np

try:
    from nes_original import sim
except ImportError:  # numba is optional; see requirements-nes-search.txt
    sim = None

from nes.actions import ActionRules,enumerate_actions,execute
from nes.engine import Game,Phase,Rules


def active_states(seeds=range(3),steps=25,reaction=3):
    rules = ActionRules(horizontal_period=2,reaction_frames=reaction)
    rng = np.random.default_rng(7)
    for seed in seeds:
        game = Game(seed,Rules(start_level=18))
        for _ in range(steps):
            while game.phase!=Phase.ACTIVE and not game.over:
                game.tick(0)
            if game.over:
                break
            observation = game.observe()
            actions = enumerate_actions(observation,rules)
            yield observation,actions
            execute(game,actions,int(rng.integers(len(actions.boards))))


@unittest.skipIf(sim is None,'numba not installed')
class FastSimulatorTest(unittest.TestCase):
    def test_candidates_match_engine_action_family(self):
        for reaction in (3,0):
            with self.subTest(reaction=reaction):
                self._check_candidates(reaction)

    def _check_candidates(self,reaction):
        checked = 0
        for observation,actions in active_states(reaction=reaction):
            board = observation.board!=0
            states = np.zeros((sim.MAX_CANDIDATES,9),np.int64)
            n = sim.candidates(board,tuple(int(v) for v in vars(observation.control).values()),
                observation.level,2,states,reaction)
            found = {}
            for i in range(n):
                result,cleared,dead = sim.lock(board,*map(int,states[i,:3]))
                found[(result.tobytes(),int(states[i,4]),bool(dead))] = cleared
            expected = {((b!=0).tobytes(),c.das,bool(t)):int(l)
                for b,l,t,c in zip(actions.boards,actions.lines,actions.terminal,actions.controls)}
            self.assertEqual(found,expected)
            checked += 1
        self.assertGreater(checked,40)

    def test_dellacherie_weights_reproduce_retained_selector(self):
        from nes_original.dellacherie import Dellacherie
        reference = Dellacherie()
        for observation,actions in active_states(seeds=(5,),steps=20):
            plan = SimpleNamespace(actions=actions)
            expected = reference.score_actions(observation,plan)
            ours = sim.action_features(observation,plan,well=-1)@sim.DELLACHERIE
            alive = ~actions.terminal
            np.testing.assert_allclose(ours[alive],expected[alive],atol=1e-9)

    def test_board_features_hand_counted(self):
        board = np.zeros((20,10),bool)
        board[16:,:9] = True       # four rows ready for a column-9 tetris
        board[15,0] = True
        board[17,4] = False        # one hole under three filled cells
        board[16,4] = True
        out = np.zeros(sim.N_FEATURES)
        sim.board_features(board,9,out)
        f = dict(zip(sim.FEATURES,out))
        self.assertEqual(f['holes'],1)
        self.assertEqual(f['covered'],1)
        self.assertEqual(f['well_height'],0)
        self.assertEqual(f['tetris_ready'],2)  # row 17 is no longer full
        board[17,4] = True
        sim.board_features(board,9,out)
        self.assertEqual(dict(zip(sim.FEATURES,out))['tetris_ready'],4)

    def test_play_is_deterministic_and_switches_mode(self):
        a = sim.play_game(seed=3,max_lines=40,beam=2)
        self.assertEqual(a,sim.play_game(seed=3,max_lines=40,beam=2))
        self.assertGreaterEqual(a['lines'],0)
        capped = sim.play_game(sim.SCORE_START,sim.DELLACHERIE,seed=3,max_lines=30,initial_score=sim.MAXED,
            well_survive=-1)
        same = sim.play_game(sim.DELLACHERIE,sim.DELLACHERIE,seed=3,max_lines=30,well=-1,well_survive=-1)
        self.assertEqual(capped['lines'],same['lines'])
        self.assertEqual(capped['placements'],same['placements'])

    def test_selector_modes_and_beam(self):
        selector = sim.TunedSelector(sim.SCORE_START,sim.DELLACHERIE,well=9,well_survive=-1,beam=3,switch_level=29)
        for observation,actions in active_states(seeds=(2,),steps=6):
            plan = SimpleNamespace(actions=actions)
            index = selector(observation,plan)
            self.assertTrue(0<=index<len(actions.boards))
        self.assertEqual(selector.modes['survive'],0)
        late = SimpleNamespace(**{**vars(observation),'level':29})
        selector(late,plan)
        self.assertEqual(selector.modes['survive'],1)
        selector(observation,plan)  # latched after the switch
        self.assertEqual(selector.modes['survive'],2)

    def test_zone_weights_apply_only_in_their_levels(self):
        selector = sim.TunedSelector(sim.DELLACHERIE,sim.DELLACHERIE,well=-1,well_survive=-1,
            zones=[(156,158,sim.SCORE_START,9)])
        for observation,actions in active_states(seeds=(2,),steps=1):
            plan = SimpleNamespace(actions=actions)
            inside = SimpleNamespace(**{**vars(observation),'level':157})
            np.testing.assert_allclose(selector.score_actions(inside,plan),
                sim.action_features(inside,plan,9)@sim.SCORE_START)
            self.assertEqual(selector.modes['zone'],1)
            outside = SimpleNamespace(**{**vars(observation),'level':159})
            np.testing.assert_allclose(selector.score_actions(outside,plan),
                sim.action_features(outside,plan,-1)@sim.DELLACHERIE)


if __name__=='__main__':
    unittest.main()


def _plan(board, level):
    states = np.zeros((sim.MAX_CANDIDATES,9),np.int64)
    state,_ = sim.spawn_state(int(sim.SPAWNS[6]),0,board,min(level,29))
    n = sim.candidates(board,state,min(level,29),2,states,3)
    boards,lines,dead,controls = [],[],[],[]
    for i in range(n):
        result,c,d = sim.lock(board,int(states[i,0]),int(states[i,1]),int(states[i,2]))
        boards.append(result.astype(np.uint8)); lines.append(c); dead.append(d)
        controls.append(SimpleNamespace(x=int(states[i,0]),y=int(states[i,1]),orientation=int(states[i,2]),
                                        das=int(states[i,4]),soft_points=int(states[i,6])))
    return SimpleNamespace(actions=SimpleNamespace(boards=boards,lines=np.asarray(lines),
        terminal=np.asarray(dead),controls=controls))


@unittest.skipIf(sim is None,'numba not installed')
class AltModeTest(unittest.TestCase):
    def test_tetris_mode_only_low_in_range_and_outside_zones(self):
        selector = sim.TunedSelector(sim.DELLACHERIE,sim.DELLACHERIE,well=-1,well_survive=-1,beam=0,
            zones=[(156,158,sim.DELLACHERIE,-1)],
            alt=(sim.SCORE_START,6,9,((29,153),),sim.DELLACHERIE))
        low = np.zeros((20,10),np.bool_); low[17:,:9] = True
        high = np.zeros((20,10),np.bool_); high[10:,:9] = True
        for board,level,mode in ((low,100,'alt'),(high,100,'burn'),(low,157,'zone'),(low,200,None)):
            with self.subTest(level=level,mode=mode):
                before = dict(selector.modes)
                obs = SimpleNamespace(board=board.astype(np.uint8),level=level,lines=10*level,score=999999,
                                      placements=0,next_orientation=int(sim.SPAWNS[2]))
                selector.score_actions(obs,_plan(board,level))
                changed = {k for k in selector.modes if selector.modes[k]!=before.get(k,0)}-{'score','survive'}
                self.assertEqual(changed,set() if mode is None else {mode})
