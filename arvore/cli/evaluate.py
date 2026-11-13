"""Evaluate a fitted Box against labelled data."""

from argparse import Namespace
import json
from pathlib import Path

from carabiner.cliutils import clicommand

from ..box import Box
from ..box.base import DEFAULT_BATCH_SIZE
from ..config.instantiate import apply_overrides


@clicommand(message="Evaluating")
def _evaluate(args: Namespace) -> None:
    from providence import DataResolver
    from providence.io import load_json

    config = (
        load_json(args.config)
        if args.config is not None
        else {}
    )
    config = apply_overrides(config, args.set)

    box = Box.load(
        args.input_file,
        cache_dir=args.cache,
        trusted_types=(
            ("sklearn.tree._tree.Tree",)
            if args.trust_sklearn_trees
            else ()
        ),
    )

    metrics = (
        args.metrics
        if args.metrics is not None
        else config.get("metrics")
    )

    if args.data is not None:
        resolved = (
            DataResolver(args.data)
            .select_rows(args.start, args.stop)
        )
        data = resolved.data

    else:
        if args.start is not None or args.stop is not None:
            raise ValueError(
                "--start and --stop require --data"
            )
        data = None

    predicted, scores = box.evaluate(
        data,
        pipeline=args.pipeline,
        metrics=metrics,
        batch_size=config.get(
            "batch_size",
            DEFAULT_BATCH_SIZE,
        ),
    )

    # Convert NumPy scalar/array results to JSON-compatible values.
    def serializable(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(
            f"Cannot serialize {type(value).__name__}"
        )

    output = json.dumps(
        scores,
        indent=2,
        default=serializable,
        allow_nan=False,
    )

    if args.result is not None:
        path = Path(args.result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    if args.output is not None:
        DataResolver(
            predicted,
            cache_dir=args.cache,
        ).write_table(args.output)
