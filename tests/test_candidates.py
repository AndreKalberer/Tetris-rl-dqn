import numpy as np
import pytest
import torch

from dqn.candidates import enumerate_candidates, enumerate_visible, preview, preview_rotation, rotations
from dqn.lookahead import choose_lookahead
from dqn.episode_stats import EpisodeStats
from dqn.train_candidates import Config, Trainer, double_targets, load_policy, make_env, values
from tetris_macro_env import _locked_playfield_01, compute_features


@pytest.fixture
def env():
    e = make_env(Config())
    e.reset(seed=0)
    yield e
    e.close()


def test_zero_seed_and_five_piece_queue(env):
    observations = [env.reset(seed=0)[0] for _ in range(3)]
    assert all(np.array_equal(observations[0], o) for o in observations)
    assert len(env.unwrapped.queue.get_queue()) == 5
    assert all(p >= 0 for p in env._next_piece_indices())
    assert observations[0][239:246].sum() == 1
    assert env.unwrapped.observation_space.contains(env.unwrapped._get_obs())


def test_locked_board_does_not_erase_blocks_inside_spawn_rectangle(env):
    u = env.unwrapped
    ys, xs = np.nonzero(u.active_tetromino.matrix == 0)
    for y, x in zip(ys, xs):
        if u.padding <= u.x+x < u.padding+u.width:
            u.board[u.y+y, u.x+x] = 2
            break
    assert _locked_playfield_01(env, u._get_obs()).sum() == 1


@pytest.mark.parametrize('hold', [False, True])
def test_invalid_request_does_not_consume_or_swap_piece(env, hold):
    before, _ = env.reset(seed=0)
    state = env.unwrapped.get_state()
    _, reward, done, trunc, info = env.step((60 if hold else 0))
    assert info['invalid_macro_action'] and not info['piece_placed']
    assert not done and not trunc and reward < 0
    assert np.array_equal(before, env._obs_from_raw(env.unwrapped._get_obs()))
    assert np.array_equal(state.board, env.unwrapped.board)
    assert state.queue.get_queue() == env.unwrapped.queue.get_queue()
    assert not env.unwrapped.has_swapped
    assert not env.unwrapped.holder.get_tetrominoes()


def test_out_of_range_action_rejected(env):
    for action in [-1, env.action_space.n]:
        with pytest.raises(ValueError):
            env.step(action)


def test_previews_match_engine_and_do_not_change_state(env):
    rng = np.random.default_rng(12)
    for _ in range(30):
        original = env.unwrapped.get_state()
        held = env._held_piece_idx
        previous = [env._prev_agg_h, env._prev_holes, env._prev_bump, env._prev_max_h]
        cs = enumerate_candidates(env)
        assert original.queue.get_queue() == env.unwrapped.queue.get_queue()
        assert np.array_equal(original.board, env.unwrapped.board)
        # Every preview, including hold and line clears, must match real execution.
        for i, action in enumerate(cs.actions):
            env.unwrapped.set_state(original)
            # get_state returns independent containers, but set_state takes ownership.
            env.unwrapped.set_state(env.unwrapped.get_state())
            env._held_piece_idx = held
            (env._prev_agg_h, env._prev_holes, env._prev_bump, env._prev_max_h) = previous
            predicted_reward = env.placement_reward(previous, compute_features(cs.boards[i]),
                int(cs.lines[i]), bool(cs.terminal[i]))
            _, actual_reward, done, _, info = env.step(int(action))
            assert not info['invalid_macro_action']
            assert np.array_equal(cs.boards[i], _locked_playfield_01(env, {})), (i, action)
            assert cs.lines[i] == info['lines_cleared']
            assert cs.terminal[i] == done
            assert predicted_reward == actual_reward
        env.unwrapped.set_state(original)
        env._held_piece_idx = held
        (env._prev_agg_h, env._prev_holes, env._prev_bump, env._prev_max_h) = previous
        _, _, done, _, _ = env.step(int(rng.choice(cs.actions)))
        if done:
            env.reset()


def test_hidden_queue_does_not_affect_features(env):
    # Keep the first two visible pieces but change queue entries behind the window.
    env.next_n = 2
    before = enumerate_candidates(env)
    u = env.unwrapped
    u.queue.queue[4] = (int(u.queue.queue[4])+1) % 7
    u.randomizer.rng = np.random.default_rng(999)
    after = enumerate_candidates(env)
    assert np.array_equal(before.actions, after.actions)
    assert np.array_equal(before.features, after.features)


def test_four_line_clear_preview_matches_engine(env):
    u = env.unwrapped
    u.active_tetromino = u.tetrominoes[0]  # I piece
    u.board[u.height-4:u.height, u.padding:u.padding+u.width] = 2
    u.board[u.height-4:u.height, u.padding+4] = 0
    cs = enumerate_candidates(env)
    i = int(cs.lines.argmax())
    assert cs.lines[i] == 4
    _, _, _, _, info = env.step(int(cs.actions[i]))
    assert info['lines_cleared'] == 4
    assert np.array_equal(cs.boards[i], _locked_playfield_01(env, {}))


def test_losing_placements_remain_available_and_end_game(env):
    u = env.unwrapped
    u.board[:u.height, u.padding:u.padding+u.width] = 2
    cs = enumerate_candidates(env)
    assert len(cs.actions) > 0 and cs.terminal.all()
    _, _, done, _, info = env.step(int(cs.actions[0]))
    assert done and not info['invalid_macro_action']
    with pytest.raises(RuntimeError):
        env.step(int(cs.actions[0]))


def test_second_ply_matches_actual_next_placements_without_hidden_draws(env):
    first = enumerate_candidates(env)
    # Check both ordinary placement and initially empty hold consumption.
    for i in [0, next(i for i,a in enumerate(first.actions) if a >= 60)]:
        env.reset(seed=0)
        children = enumerate_visible(first.states[i])
        env.step(int(first.actions[i]))
        actual = enumerate_candidates(env)
        assert np.array_equal(children.actions, actual.actions)
        assert np.array_equal(children.boards, actual.boards)
        assert np.array_equal(children.lines, actual.lines)
        assert np.array_equal(children.terminal, actual.terminal)
        assert -1 in first.states[i].queue  # Newly drawn hidden pieces stay unknown.


def test_two_ply_choice_is_legal_and_does_not_advance_engine(env):
    cs = enumerate_candidates(env)
    model = torch.nn.Linear(cs.features.shape[1], 1)
    before = env.unwrapped.get_state()
    idx, metrics = choose_lookahead(env, cs, model, beam=4)
    assert 0 <= idx < len(cs.actions) and metrics['evaluated_second_moves'] > 0
    assert np.array_equal(before.board, env.unwrapped.board)
    assert before.queue.get_queue() == env.unwrapped.queue.get_queue()
    plain, _ = choose_lookahead(env, cs, model, beam=1)
    assert plain == int(values(model, cs.features).argmax())


def test_batched_drop_matches_reference_across_random_cavities_and_rotations():
    rng = np.random.default_rng(123)
    for _ in range(30):
        board = (rng.random((20, 10)) < .35).astype(np.uint8)
        board[:int(rng.integers(0, 21))] = 0
        for piece in range(7):
            for _, ys, xs in rotations(piece):
                positions = np.arange(-int(xs.min()), 10-int(xs.max()))
                boards, lines, terminal = preview_rotation(board, ys, xs, positions)
                for i, x in enumerate(positions):
                    reference = preview(board, ys, xs, int(x))
                    assert np.array_equal(boards[i], reference[0])
                    assert lines[i] == reference[1] and terminal[i] == reference[2]


def test_partial_episode_excluded_from_mean():
    stats = EpisodeStats()
    stats.add(10, True)
    stats.add(2, False)
    assert stats.mean_lines == 10 and stats.partial_lines == 2
    assert stats.mean_steps == 1 and stats.partial_steps == 1
    assert EpisodeStats().mean_lines is None


def test_double_target_uses_online_choice_target_score_and_terminal_mask():
    online = torch.nn.Linear(1, 1, bias=False)
    target = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        online.weight.fill_(1)
        target.weight.fill_(-2)
    batch = [(np.array([0]), 1, np.array([[1], [3]], dtype=np.float16), False),
             (np.array([0]), -5, np.empty((0, 1)), True)]
    assert torch.allclose(double_targets(online, target, batch, .5), torch.tensor([-2., -5.]))


def test_checkpoint_continues_same_training_trajectory(tmp_path):
    config = Config(seed=0, capacity=32, batch_size=4, learning_starts=4,
                    target_every=4, threads=1)
    first = Trainer(config)
    for _ in range(12):
        first.advance()
    path = tmp_path/'resume.pt'
    first.save(path)
    first.export_policy(tmp_path/'policy.pt')
    config_loaded, policy = load_policy(tmp_path/'policy.pt')
    assert config_loaded == config
    assert np.array_equal(values(first.online, first.candidates.features), values(policy, first.candidates.features))
    resumed = Trainer.load(path)
    for _ in range(8):
        first.advance()
        resumed.advance()
    assert first.step == resumed.step == 20
    assert len(first.replay.items) == len(resumed.replay.items)
    assert np.array_equal(first.env.unwrapped.board, resumed.env.unwrapped.board)
    assert first.last_loss == resumed.last_loss
    for a, b in zip(first.online.parameters(), resumed.online.parameters()):
        assert torch.equal(a, b)
    first.env.close()
    resumed.env.close()
