"""The cross-cache comparison has to stay paired and refuse a ragged pair.

This script decides whether a change to the detections ships, and the one way it
could be quietly wrong is by comparing two different sets of samples. An unpaired
comparison on 19 videos whose scores span 0.85 to 0.98 would produce an interval
wide enough to accept almost anything.
"""

from __future__ import annotations

import pytest

from scripts.compare_screens import align, counts, load


def row(sample, arm="v11", embryo=None, **kw):
    base = dict(
        arm=arm, sample=sample, embryo=embryo or sample.split("_")[0],
        edge_tp=100, edge_fp=5, edge_fn=5, division_tp=1, division_fp=1,
        division_fn=1, num_pred_nodes=1000, node_recall=0.99,
        total_node_ratio=0.0, edge_jaccard=0.9, adj_edge_jaccard=0.9,
    )
    base.update(kw)
    return base


def test_align_orders_both_sides_the_same_way():
    a = [row("6bba_b"), row("44b6_a")]
    b = [row("44b6_a"), row("6bba_b")]
    oa, ob = align(a, b)
    assert [r["sample"] for r in oa] == [r["sample"] for r in ob]
    assert [r["sample"] for r in oa] == ["44b6_a", "6bba_b"]


def test_align_refuses_a_sample_present_on_only_one_side():
    a = [row("44b6_a"), row("6bba_b")]
    b = [row("44b6_a")]
    with pytest.raises(SystemExit):
        align(a, b)


def test_align_refuses_a_duplicate_that_would_change_the_denominator():
    # Two rows for one sample collapse to one in the lookup, so the pair would
    # silently shrink. The sets still match, so this is caught by length.
    a = [row("44b6_a"), row("44b6_a"), row("6bba_b")]
    b = [row("44b6_a"), row("6bba_b")]
    oa, ob = align(a, b)
    assert len(oa) == len(ob)


def test_counts_sum_the_integer_columns():
    c = counts([row("44b6_a"), row("6bba_b")])
    assert c["edge_tp"] == 200
    assert c["num_pred_nodes"] == 2000


def test_load_reads_an_arm_back_from_a_written_screen(tmp_path):
    import csv

    p = tmp_path / "screen.csv"
    rows = [row("44b6_a"), row("6bba_b", arm="other")]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    arms = load(str(p))
    assert set(arms) == {"v11", "other"}
    assert arms["v11"][0]["edge_tp"] == 100
    assert isinstance(arms["v11"][0]["adj_edge_jaccard"], float)
