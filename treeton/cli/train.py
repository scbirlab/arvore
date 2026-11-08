"""Train a Box from declarative configuration."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box import Box
from ..config import resolve_experiment_config


@clicommand(message="Training model")
def _train(args: Namespace) -> None:
    from providence.io import load_json
    config = resolve_experiment_config(
        load_json(args.config),
        model=args.model,
        overrides=args.set,
    )
    try:
        box_config = config["box"]
    except KeyError as error:
        raise ValueError(
            "Training config requires "
            "a `box` section."
        ) from error

    box = Box.from_config(
        box_config,
        cache_dir=args.cache,
    )
    fit_kwargs = dict(config.get("fit", {}))
    box.fit(
        args.input_file,
        validation=args.validation,
        **fit_kwargs,
    )
    save_kwargs = dict(config.get("save", {}))
    box.save(
        args.output,
        **save_kwargs,
    )
