"""Bounded two-ply inference using only observed queue pieces.

The frozen network proposes the top K first moves. Each is rescored using its
exact immediate shaped reward plus discounted best second-move network Q. Unknown
new queue entries stay unknown (-1); no simulator RNG or hidden pieces are read.
"""
import numpy as np

from dqn.candidates import enumerate_visible
from tetris_macro_env import compute_features


def choose_lookahead(env, candidates, model, *, gamma=.99, beam=4, board_input=True):
    from dqn.train_candidates import values
    if beam < 1:
        raise ValueError('Beam must be positive')
    initial = values(model, candidates.features)
    if beam == 1:
        return int(initial.argmax()), dict(expanded_first_moves=0, evaluated_second_moves=0)
    # Stable ordering makes ties reproducible and preserves the original argmax.
    indices = np.argsort(-initial, kind='stable')[:beam]
    previous = (env._prev_agg_h, env._prev_holes, env._prev_bump, env._prev_max_h)
    scores, child_groups = {}, []
    for i in indices:
        i = int(i)
        scores[i] = env.placement_reward(previous, compute_features(candidates.boards[i]),
                                         int(candidates.lines[i]), bool(candidates.terminal[i]))
        if not candidates.terminal[i]:
            children = enumerate_visible(candidates.states[i], board_input=board_input)
            child_groups.append((i, children.features))
    count = sum(len(fs) for _, fs in child_groups)
    if child_groups:
        packed = np.concatenate([fs for _, fs in child_groups])
        q = values(model, packed)
        offset = 0
        for i, fs in child_groups:
            scores[i] += gamma*float(q[offset:offset+len(fs)].max())
            offset += len(fs)
    best = max((int(i) for i in indices), key=lambda i: scores[i])
    return best, dict(expanded_first_moves=len(indices), evaluated_second_moves=count)
