import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from spd.app.backend.lib.activation_contexts import _apply_position_separation


def test_separation():
    seq_positions = np.array([10, 12, 16], dtype=np.int64)
    batch_indices = np.array([0, 0, 0], dtype=np.int64)
    min_separation = 5

    print(f"Testing with positions {seq_positions} and min_separation {min_separation}")
    keep = _apply_position_separation(seq_positions, batch_indices, min_separation)
    print(f"Keep mask: {keep}")

    expected = np.array([True, False, True])
    if np.array_equal(keep, expected):
        print("Result matches EXPECTED behavior (Code is CORRECT).")
    else:
        print("Result matches BUGGY behavior (Code is INCORRECT).")
        # If buggy, we expect [True, False, False]
        buggy_expected = np.array([True, False, False])
        if np.array_equal(keep, buggy_expected):
            print("Confirmed BUGGY behavior.")


if __name__ == "__main__":
    test_separation()
