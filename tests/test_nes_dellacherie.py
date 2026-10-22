"""Hand-counted feature fixtures; no claim of original-game playing strength."""
import numpy as np

from nes.engine import Control
from nes_original.dellacherie import board_features, placement_features


def test_empty_board_has_only_sidewall_and_floor_transitions():
    assert np.array_equal(board_features(np.zeros((20, 10), bool)), [40, 10, 0, 0])


def test_buried_empty_cell_and_two_deep_well_count_separately():
    board = np.zeros((20, 10), bool)
    board[18:, :] = True
    board[18:, 4] = False
    assert board_features(board)[3] == 3  # depth1+depth2, open from above
    assert board_features(board)[2] == 0
    board[18, 4] = True
    assert board_features(board)[2] == 1
    assert board_features(board)[3] == 1


def test_eroded_cells_credits_only_added_piece_cells_in_cleared_rows():
    # Place the vertical I in the last-column gap.
    from nes.engine import CELLS, TYPE
    vertical = next(i for i, cells in enumerate(CELLS)
                    if TYPE[i] == 6 and len({dx for dx, dy in cells}) == 1)
    cells = CELLS[vertical]
    x = 9-cells[0][0]
    y = 19-max(dy for dx, dy in cells)
    before = np.zeros((20, 10), bool)
    before[16:, :9] = True
    state = Control(x=x, y=y, orientation=vertical)
    result = placement_features(before, state, np.zeros((20, 10), bool), 4)
    assert result[0] == 2.5
    assert result[1] == 16
    assert np.array_equal(result[2:], [40, 10, 0, 0])
