"""The learned linker, tested on affinities we construct rather than predict.

The point of these is that the solver behaviour is pinned independently of the
network. If the score moves when the edge head is wired in, it should be because
the edge head disagrees with distance, not because the assignment silently
changed shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.link import (
    add_divisions,
    link_sequence,
    link_sequence_learned,
    match_consecutive,
    match_consecutive_learned,
)

SCALE = (1.625, 0.40625, 0.40625)


def aff(pairs: list[tuple[int, int, float]]) -> dict:
    return {
        "i": np.asarray([p[0] for p in pairs], dtype=np.int32),
        "j": np.asarray([p[1] for p in pairs], dtype=np.int32),
        "p": np.asarray([p[2] for p in pairs], dtype=np.float32),
    }


def test_picks_the_higher_probability_pairing():
    # Two ways to pair two nodes with two. The crossed pairing is likelier, and
    # a distance linker on these coordinates would pick the other one.
    a = aff([(0, 0, 0.2), (0, 1, 0.9), (1, 0, 0.9), (1, 1, 0.2)])
    assert sorted(match_consecutive_learned(2, 2, a)) == [(0, 1), (1, 0)]


def test_maximises_the_product_not_the_sum():
    # Sum of probabilities prefers 0.99 + 0.01 = 1.00 over 0.60 + 0.60 = 1.20?
    # No: sum prefers the pair of 0.6s only because 1.2 > 1.0. Build the case
    # where the two rules disagree, 0.99 + 0.01 against 0.50 + 0.50, sum 1.00
    # both ways, product 0.0099 against 0.2500.
    a = aff([(0, 0, 0.99), (1, 1, 0.01), (0, 1, 0.50), (1, 0, 0.50)])
    assert sorted(match_consecutive_learned(2, 2, a)) == [(0, 1), (1, 0)]


def test_a_pair_never_proposed_is_never_linked():
    # Node 1 has no candidate at all. It must end its track rather than be
    # forced into the only remaining slot.
    a = aff([(0, 0, 0.9)])
    assert match_consecutive_learned(2, 2, a) == [(0, 0)]


def test_empty_inputs_are_not_errors():
    assert match_consecutive_learned(0, 3, aff([])) == []
    assert match_consecutive_learned(3, 0, aff([])) == []
    assert match_consecutive_learned(3, 3, aff([])) == []


def test_zero_probability_does_not_produce_a_non_finite_cost():
    # -log(0) is inf, and an inf in the cost matrix takes scipy's solver with it.
    a = aff([(0, 0, 0.0), (1, 1, 0.5)])
    got = match_consecutive_learned(2, 2, a)
    assert (1, 1) in got


def test_out_degree_stays_at_one_without_divisions():
    a = aff([(0, 0, 0.9), (0, 1, 0.8)])
    got = match_consecutive_learned(1, 2, a)
    assert len(got) == 1


def test_graph_matches_the_distance_linker_when_affinity_agrees_with_distance():
    # Same detections, and an affinity that ranks pairs exactly as distance does.
    # The two linkers must then produce the same edges, which is what makes a
    # difference in a real run attributable to the edge head.
    rng = np.random.default_rng(0)
    frames = [rng.uniform(0, 20, size=(6, 3)) for _ in range(4)]
    affinities = []
    for t in range(3):
        a, b = frames[t], frames[t + 1]
        pa = a * np.asarray(SCALE)[None, :]
        pb = b * np.asarray(SCALE)[None, :]
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        pairs = []
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                if d[i, j] <= 7.0:
                    # Monotone decreasing in distance, and the assignment sums
                    # -log(p), so this recovers the distance objective exactly.
                    pairs.append((i, j, float(np.exp(-d[i, j]))))
        affinities.append(aff(pairs))

    g_dist = link_sequence(frames, SCALE, max_link_um=7.0)
    g_learn = link_sequence_learned(frames, affinities, SCALE)
    assert np.array_equal(
        np.sort(g_dist.edges, axis=0), np.sort(g_learn.edges, axis=0)
    )


def test_wrong_number_of_affinity_entries_is_an_error():
    frames = [np.zeros((2, 3)) for _ in range(4)]
    with pytest.raises(ValueError, match="affinity entries"):
        link_sequence_learned(frames, [aff([])], SCALE)


def test_node_ids_and_frame_offsets_match_the_distance_linker():
    frames = [np.zeros((3, 3)), np.zeros((2, 3)), np.zeros((4, 3))]
    affinities = [aff([]), aff([])]
    g = link_sequence_learned(frames, affinities, SCALE)
    assert len(g.nodes) == 9
    assert list(g.nodes.t) == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert g.edges.shape == (0, 2)
