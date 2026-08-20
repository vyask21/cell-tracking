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

    # And the fold must refuse to run if it is not actually embryo-disjoint,
    # since the reference trainer's fallback is a seeded 90/10 over all 199.
    assert "not embryo-disjoint" in code
    assert "raise SystemExit" in code
