"""CLI scorer expression parsing and configuration tests."""

import pytest

from treeton.cli.scoring import parse_scorer, resolve_cli_scores
from treeton.scorers import (
    EnsembleSD,
    FeatureDistance,
    TanimotoDistance,
    TreeSD,
)


def test_existing_registered_names():
    resolved = resolve_cli_scores(
        names=["ensemble_sd", "tree_sd", "feature_distance"]
    )
    assert isinstance(resolved["ensemble_sd"], EnsembleSD)
    assert isinstance(resolved["tree_sd"], TreeSD)
    assert isinstance(resolved["feature_distance"], FeatureDistance)


def test_short_aliases():
    scores = resolve_cli_scores(names=[
        "tanimoto(n_neighbors=2)",
        "feature(metric='manhattan',n_neighbors=4)",
    ])
    assert isinstance(scores["tanimoto"], TanimotoDistance)
    assert scores["tanimoto"].n_neighbors == 2
    assert scores["feature"].metric == "manhattan"
    assert scores["feature"].n_neighbors == 4


def test_multiple_named_instances():
    scores = resolve_cli_scores(names=[
        "tan2=tanimoto(n_neighbors=2)",
        "tan8=tanimoto(n_neighbors=8,return_var=True)",
    ])
    assert scores["tan2"].n_neighbors == 2
    assert scores["tan8"].n_neighbors == 8
    assert scores["tan8"].return_var is True


def test_literal_types():
    scorer = parse_scorer(
        "tanimoto("
        "n_neighbors=2,"
        "return_var=True,"
        "return_neighbors=False,"
        "neighbor_delimiter='|')"
    )
    assert scorer.n_neighbors == 2
    assert scorer.return_var is True
    assert scorer.return_neighbors is False
    assert scorer.neighbor_delimiter == "|"


def test_config_strings_and_lists():
    single = resolve_cli_scores(
        config="tanimoto(n_neighbors=3)"
    )
    assert single["tanimoto"].n_neighbors == 3

    multiple = resolve_cli_scores(
        config=[
            "ensemble_sd",
            "fast=feature(n_neighbors=2)",
        ]
    )
    assert isinstance(multiple["ensemble_sd"], EnsembleSD)
    assert multiple["fast"].n_neighbors == 2


def test_existing_class_path_configuration():
    config = {
        "novelty": {
            "class_path": "treeton.scorers.tanimoto.TanimotoDistance",
            "init_args": {
                "n_neighbors": 4,
                "return_var": True,
            },
        },
    }
    scores = resolve_cli_scores(config=config)
    assert isinstance(scores["novelty"], TanimotoDistance)
    assert scores["novelty"].n_neighbors == 4
    assert scores["novelty"].return_var


def test_config_and_cli_merge():
    scores = resolve_cli_scores(
        config={"ensemble": "ensemble_sd"},
        names=["tan=tanimoto(n_neighbors=2)"],
    )
    assert list(scores) == ["ensemble", "tan"]


def test_duplicate_outputs_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_cli_scores(
            config={"tan": "tanimoto_distance"},
            names=["tan=tanimoto(n_neighbors=2)"],
        )

    with pytest.raises(ValueError, match="Duplicate"):
        resolve_cli_scores(
            names=[
                "tanimoto(n_neighbors=2)",
                "tanimoto(n_neighbors=4)",
            ],
        )


@pytest.mark.parametrize("expression", [
    "tanimoto(n_neighbors=missing)",
    "tanimoto(n_neighbors=__import__('os'))",
    "tanimoto(n_neighbors=(1).__class__)",
    "tanimoto(*[2])",
    "tanimoto(**{'n_neighbors': 2})",
    "tanimoto(2)",
    "tanimoto(n_neighbors=2); print('bad')",
    "__import__('os')",
])
def test_unsafe_or_invalid_expressions_rejected(expression):
    with pytest.raises(ValueError):
        parse_scorer(expression)


def test_unknown_scorer_and_argument():
    with pytest.raises(ValueError, match="Unknown scorer"):
        parse_scorer("unknown(n_neighbors=2)")
    with pytest.raises(ValueError, match="Invalid arguments"):
        parse_scorer("tree_sd(n_neighbors=2)")


def test_no_scores():
    assert resolve_cli_scores() is None
