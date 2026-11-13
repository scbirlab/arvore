"""Train a Box from declarative configuration."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box import Box
from ..config.instantiate import (
    load_training_config,
    resolve_experiment_config,
)


@clicommand(message="Training model")
def _train(args: Namespace) -> None:
    config = resolve_experiment_config(
        load_training_config(args.config),
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

    if box_config.get("model") is None:
        raise ValueError(
            "This checkpoint has no reconstructible model configuration. "
            "Supply --model or --set box.model=... to retrain."
        )

    box = Box.from_config(
        box_config,
        cache_dir=args.cache,
    )
    box.fit(
        args.input_file,
        validation=args.validation,
        **config.get("fit", {}),
    )
    # Train-config save.score uses the same declarative scorer syntax as
    # prediction config; Box.save accepts the resolved scorer instances.
    from .scoring import resolve_cli_scores
    save_kwargs = dict(config.get("save", {}))

    if args.skip_ref_save:
        # The CLI opt-out overrides save.save_references in JSON.
        # No need to instantiate or prepare configured scorers.
        save_kwargs.pop("score", None)
        save_kwargs["save_references"] = False

    elif save_kwargs.get("save_references", True):
        if "score" in save_kwargs:
            save_kwargs["score"] = resolve_cli_scores(
                config=save_kwargs["score"],
            )

    else:
        # Declarative opt-out has the same semantics as the CLI flag.
        save_kwargs.pop("score", None)

    box.save(args.checkpoint, **save_kwargs)
