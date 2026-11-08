"""Predict from a Box checkpoint."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box import Box, DEFAULT_BATCH_SIZE
from ..config import apply_overrides


@clicommand(message="Predicting")
def _predict(args: Namespace) -> None:

    from providence.io import load_json
    from providence import DataResolver

    box = Box.load(
        args.checkpoint,
        cache_dir=args.cache,
    )
    config = (
        load_json(args.config)
        if args.config is not None
        else {}
    )
    config = apply_overrides(config, args.set)

    result = box.predict(
        args.data,
        batch_size=config.get(
            "batch_size",
            DEFAULT_BATCH_SIZE,
        ),
    )
    resolved = DataResolver(result, cache_dir=args.cache_dir)
    resolved.save(args.output)
