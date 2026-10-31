"""Train a Box from declarative configuration."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box import Box
from ..checkpoint_utils import load_json
from ..config import (
    instantiate_trainer,
    resolve_experiment_config,
)


@clicommand(message="Training model")
def _train(args: Namespace) -> None:

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

    try:
        trainer_config = config["trainer"]
    except KeyError as error:
        raise ValueError(
            "Training config requires "
            "a `trainer` section."
        ) from error

    box = Box.from_config(
        box_config,
        cache_dir=args.cache,
    )
    box.trainer = instantiate_trainer(trainer_config)

    fit_kwargs = dict(config.get("fit", {}))

    box.fit(
        args.training,
        validation=args.validation,
        **fit_kwargs,
    )

    derivatives = config.get("derivatives")
    if derivatives:
        box.compute_training_derivatives(**derivatives)

    save_kwargs = dict(config.get("save", {}))
    box.save(
        args.output,
        **save_kwargs,
    )
