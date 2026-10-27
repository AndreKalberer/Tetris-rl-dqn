"""The ported NES evaluator plays the offline game with legal, replayable inputs."""
from offline.compare_bots import play


def test_classic_bot_passes_a_development_sprint():
    row = play(('classic', 1000, 'sprint', 0, 8, 3, 120, 30))
    assert row['passed'] and row['replay_verified']
    assert row['rejected_inputs'] == 0 and row['path_mismatches'] == 0
