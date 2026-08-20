"""The ILP linker, on graphs small enough to know the right answer by hand.

These pin the two things our own code is responsible for: the translation into
and out of the solver's node numbering, and the guard that stops an unfiltered
result being treated as a solution. The optimisation itself is the pack's and is
not retested here.

Skipped when tracksdata is absent, since the rerun only needs it for a config
that asks for the ILP.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tracksdata")

from src.link_ilp import link_sequence_ilp

SCALE = (1.625, 0.40625, 0.40625)


def aff(pairs):
    return {
        "i": np.asarray([p[0] for p in pairs], dtype=np.int32),
        "j": np.asarray([p[1] for p in pairs], dtype=np.int32),
        "p": np.asarray([p[2] for p in pairs], dtype=np.float32),
    }


def test_picks_the_coherent_tracks_through_three_frames():
    # Two cells moving apart. The crossed links are cheap individually but
    # incoherent as tracks, which is the case a per-frame solver cannot see.
    frames = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 12.0]]),
    ]
    affs = [
        aff([(0, 0, 0.9), (0, 1, 0.05), (1, 0, 0.05), (1, 1, 0.9)]),
        aff([(0, 0, 0.9), (0, 1, 0.05), (1, 0, 0.05), (1, 1, 0.9)]),
    ]
    g = link_sequence_ilp(frames, affs, SCALE)
    got = {(int(a), int(b)) for a, b in g.edges}
    # Node ids are frame-major: frame 0 is 0,1; frame 1 is 2,3; frame 2 is 4,5.
    assert got == {(0, 2), (1, 3), (2, 4), (3, 5)}


def test_node_ids_and_count_match_the_other_linkers():
    frames = [np.zeros((3, 3)), np.zeros((2, 3)), np.zeros((4, 3))]
    g = link_sequence_ilp(frames, [aff([]), aff([])], SCALE)
    assert len(g.nodes) == 9
    assert list(g.nodes.t) == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert g.edges.shape == (0, 2)


def test_no_candidates_gives_an_empty_edge_set_not_an_error():
    frames = [np.zeros((2, 3)), np.zeros((2, 3))]
    g = link_sequence_ilp(frames, [aff([])], SCALE)
    assert g.edges.shape == (0, 2)


def test_empty_detections_are_not_an_error():
    g = link_sequence_ilp([np.empty((0, 3)), np.empty((0, 3))], [aff([])], SCALE)
    assert len(g.nodes) == 0
    assert g.edges.shape == (0, 2)


def test_wrong_number_of_affinity_entries_is_an_error():
    frames = [np.zeros((2, 3)) for _ in range(4)]
    with pytest.raises(ValueError, match="affinity entries"):
        link_sequence_ilp(frames, [aff([])], SCALE)


def test_selected_edges_only():
    # Every candidate is offered; a correct solve must reject some of them.
    # If the result ever comes back with all of them, the `solution` filter in
    # _edges_from_solved has stopped working and this is the test that says so.
    frames = [np.zeros((2, 3)), np.zeros((2, 3))]
    affs = [aff([(0, 0, 0.9), (0, 1, 0.8), (1, 0, 0.7), (1, 1, 0.6)])]
    g = link_sequence_ilp(frames, affs, SCALE)
    assert g.edges.shape[0] < 4


def test_edges_only_ever_step_one_frame_forward():
    rng = np.random.default_rng(0)
    frames = [rng.uniform(0, 5, size=(3, 3)) for _ in range(4)]
    affs = [aff([(i, j, 0.5 + 0.1 * ((i + j) % 3)) for i in range(3) for j in range(3)])
            for _ in range(3)]
    g = link_sequence_ilp(frames, affs, SCALE)
    t = np.asarray(g.nodes.t)
    for a, b in g.edges:
        assert t[int(b)] - t[int(a)] == 1
