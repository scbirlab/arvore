"""Train a Box from declarative configuration."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box import Box
from ..config.instantiate import resolve_experiment_config


@clicommand(message="Training model")
def _train(args: Namespace) -> None:
    from providence.io import load_json
    config = resolve_experiment_config(
        load_json(args.config),
        model=args.model,
        overrides=args.set,
    )
    try:
        box_config = dict(config["box"])
    except KeyError as error:
        raise ValueError(
            "Training config requires "
            "a `box` section."
        ) from error
    
    if args.pipeline is not None:
        box_config["pipeline"] = args.pipeline
    box = Box.from_config(
        box_config,
        cache_dir=args.cache,
    )
    box.fit(
        args.input_file,
        validation=args.validation,
        **config.get("fit", {}),
    )
    if args.cache_leaf_reference:
        from ..scorers import InverseLeafOccupancy
        from ..box.base import DEFAULT_BATCH_SIZE

        InverseLeafOccupancy(
            normalization="reference_percentile"
        ).prepare(
            box,
            batch_size=config.get(
                "batch_size",
                DEFAULT_BATCH_SIZE,
            ),
        )

    box.save(
        args.checkpoint,
        **config.get("save", {}),
    )
