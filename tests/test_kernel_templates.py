"""The generated Kaggle scripts must parse, checked here rather than on Kaggle.

A template is a Python string that emits Python. Escapes in it are resolved when
`src/kernel.py` is parsed, not when the emitted file is, so a `\n` written in a
template becomes a real newline inside the generated source. That is invisible
locally, invisible in review, and fatal on Kaggle: the training kernel died at
line 40 with an unterminated string literal after it had already claimed a GPU
session.

Generating each template and parsing the result costs milliseconds and catches
the whole class.
"""

from __future__ import annotations

import ast

from src import kernel

COMPETITION = "biohub-cell-tracking-during-development"


def test_training_script_parses():
    for split in (0, 1):
        for max_iters_arg in ('[]', '["--max-iters", "30"]'):
            code = kernel.TRAIN_TEMPLATE.format(
                competition=COMPETITION,
                pack_slug=kernel.PACK_SLUG,
                split=split,
                epochs=3,
                batch_size=8,
                max_iters_arg=max_iters_arg,
                splits_name="dataset_splits.json",
                require_disjoint=True,
            )
            ast.parse(code)


def test_inference_script_parses_with_and_without_the_unet_prelude():
    for prelude in ("", kernel.UNET_PRELUDE):
        code = kernel.NOTEBOOK_TEMPLATE.format(
            competition=COMPETITION,
            config_name="unet50_ilp_link",
            config_hash="deadbeef",
            config_file="unet50_ilp.yaml",
            dataset_slug=kernel.DATASET_SLUG,
            archive=kernel.ARCHIVE_NAME,
            user="vyask21",
            pack_slug=kernel.PACK_SLUG,
            prelude=prelude,
        )
        ast.parse(code)


def test_the_training_script_installs_the_solver_and_never_seeds_from_the_pack():
    code = kernel.TRAIN_TEMPLATE.format(
        competition=COMPETITION, pack_slug=kernel.PACK_SLUG, split=0,
        epochs=1, batch_size=8, max_iters_arg='[]',
        splits_name="dataset_splits.json", require_disjoint=True,
    )
    assert "ilpy" in code

    # The pack trained on 180 videos from BOTH embryos, so seeding from its
    # weights would put the held-out embryo into the model before the first
    # step. That is the leak the embryo-disjoint retrain exists to remove.
    #
    # Checked against the argv list rather than the whole file, because the
    # template's docstring names the flag while explaining why it is absent.
    # A blunt substring check fails on the explanation, which is how this test
    # first went red.
    argv = code.split("sys.argv = [", 1)[1].split("]", 1)[0]
    assert "--unet-weights" not in argv
    assert "--splits" in argv and "--split" in argv

    # And the fold must refuse to run if it is not embryo-disjoint,
    # since the reference trainer's fallback is a seeded 90/10 over all 199.
    assert "should be embryo-disjoint" in code
    assert "raise SystemExit" in code


def test_the_matched_leak_run_may_share_an_embryo_but_never_a_video():
    """The leak measurement's split 1 shares an embryo on purpose.

    Split 0 trains on 6bba only and split 1 swaps 36 of those videos for 44b6
    ones, with both evaluated on the same 35 44b6 videos. Sharing the embryo is
    the variable under test, so the disjointness guard has to be switchable.
    Sharing a *video* is never acceptable and stays fatal in both.
    """
    shared = kernel.TRAIN_TEMPLATE.format(
        competition=COMPETITION, pack_slug=kernel.PACK_SLUG, split=1,
        epochs=1, batch_size=4, max_iters_arg='[]',
        splits_name="leak_splits.json", require_disjoint=False,
    )
    ast.parse(shared)
    assert "leak_splits.json" in shared
    assert "if False and overlap" in shared
    assert "in both train and test" in shared


def test_cache_script_parses():
    from src.graphcache import HELDOUT

    for views in (1, 4, 8):
        code = kernel.CACHE_TEMPLATE.format(
            competition=COMPETITION,
            dataset_slug=kernel.DATASET_SLUG,
            user="someone",
            pack_slug=kernel.PACK_SLUG,
            prelude=kernel.UNET_PRELUDE,
            n_samples=len(HELDOUT),
            det_threshold=0.99,
            det_tta=(views > 1),
            views=views,
            gate_um=20.0,
            edge_threshold=0.05,
            pool_kernel_um=5.0,
        )
        ast.parse(code)


def test_the_cache_script_settings_match_the_cache_it_is_compared_against():
    """Every setting but the TTA group has to equal the local cache's.

    `data/meta/graph_cache_t099` was built at threshold 0.99, pool 5 um, gate
    20 um and edge threshold 0.05. A screen comparing a TTA arm against it is
    only a one-variable comparison if those four agree, and a mismatch would not
    raise anywhere: it would just silently measure two changes at once.
    """
    from src.graphcache import HELDOUT

    code = kernel.CACHE_TEMPLATE.format(
        competition=COMPETITION,
        dataset_slug=kernel.DATASET_SLUG,
        user="someone",
        pack_slug=kernel.PACK_SLUG,
        prelude=kernel.UNET_PRELUDE,
        n_samples=len(HELDOUT),
        det_threshold=0.99,
        det_tta=True,
        views=8,
        gate_um=20.0,
        edge_threshold=0.05,
        pool_kernel_um=5.0,
    )
    for expected in ("det_threshold=0.99,", "pool_kernel_um=5.0,",
                     "max_link_um=20.0,", "edge_threshold=0.05,",
                     "det_tta=True,", "det_tta_views=8,"):
        assert expected in code, expected
    # The cache is worthless if it silently runs on CPU: at 6.8x a plain pass
    # that cannot finish inside the twelve hour limit, and a partial cache reads
    # as a complete one.
    assert 'device="cuda"' in code
    assert "refusing to build the cache on CPU" in code


def test_the_cache_script_covers_every_heldout_sample():
    from src.graphcache import HELDOUT

    assert len(HELDOUT) == 19
    assert len(set(HELDOUT)) == 19
    assert sum(1 for s in HELDOUT if s.startswith("44b6")) == 5
    assert sum(1 for s in HELDOUT if s.startswith("6bba")) == 14
