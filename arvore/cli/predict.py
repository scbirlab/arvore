"""Predict from a Box checkpoint."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box.base import DEFAULT_BATCH_SIZE
from ..config.instantiate import apply_overrides
from .common import load_box, selected_rows
from .scoring import resolve_cli_scores


@clicommand(message="Predicting")
def _predict(args: Namespace) -> None:
    from providence import DataResolver
    from providence.io import load_json

    if args.checkpoint is None and args.output is None:
        raise ValueError(
            "Set at least one of --checkpoint and --output"
        )

    config = (
        load_json(args.config)
        if args.config is not None
        else {}
    )
    config = apply_overrides(config, args.set)

    scorers = resolve_cli_scores(
        config=config.get("score"),
        names=args.score,
    )

    box = load_box(args)

    data = selected_rows(args)

    result = box.predict(
        data,
        pipeline=args.pipeline,
        batch_size=config.get(
            "batch_size",
            DEFAULT_BATCH_SIZE,
        ),
        score=scorers,
        disaggregate=args.disaggregate,
    )

    resolved = DataResolver(
        result,
        cache_dir=args.cache,
    )

    if args.checkpoint is not None:
        resolved.save(args.checkpoint)

    if args.output is not None:
        resolved.write_table(args.output)
