"""Predict from a Box checkpoint."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box import Box
from ..box.base import DEFAULT_BATCH_SIZE
from ..config import apply_overrides


@clicommand(message="Predicting")
def _predict(args: Namespace) -> None:

    from providence.io import load_json
    from providence import DataResolver

    if args.checkpoint is None and args.output is None:
        raise ValueError("Set at least one of --checkpoint and --output")
    
    box = Box.load(
        args.input_file,
        cache_dir=args.cache,
    )
    config = (
        load_json(args.config)
        if args.config is not None
        else {}
    )
    config = apply_overrides(config, args.set)

    resolved = DataResolver(args.data).select_rows(args.start, args.stop)

    result = box.predict(
        resolved.data,
        pipeline=args.pipeline,
        batch_size=config.get(
            "batch_size",
            DEFAULT_BATCH_SIZE,
        ),
    )
    resolved = DataResolver(result, cache_dir=args.cache)
    if args.checkpoint is not None:
        resolved.save(args.checkpoint)
    if args.output is not None:
        resolved.write_table(args.output)
