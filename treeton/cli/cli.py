"""Command-line interface for treeton."""

from carabiner.cliutils import (
    CLIApp,
    CLICommand,
    CLIOption,
)

from .. import (
    __version__,
    app_name,
)
from .predict import _predict
from .train import _train


def main() -> None:

    input_name = CLIOption(
        "input_file",
        type=str,
    )
    output_name = CLIOption(
        "--output",
        "-o",
        required=False,
        default=None,
        type=str,
        help="Where to save table",
    )
    checkpoint_out = CLIOption(
        "--checkpoint",
        "-c",
        required=False,
        default=None,
        type=str,
        help="Where to save checkpoint",
    )

    config_opt = CLIOption(
        "--config",
        required=True,
        type=str,
    )
    set_opt = CLIOption(
        "--set",
        dest="set",
        action="append",
        default=None,
        type=str,
        help=(
            "Override configuration with "
            "dotted.path=value. May be repeated."
        ),
    )
    pipeline_opt = CLIOption(
        "--pipeline",
        "-p",
        default=None,
        type=str,
        help=(
            "Pipeline definition to use to override box.pipeline."
        ),
    )
    cache_opt = CLIOption(
        "--cache",
        default=None,
        type=str,
        help="Cache directory for data loading and preprocessing.",
    )

    start = CLIOption(
        "--start",
        default=None,
        type=int,
        help=(
            "For chunking, where to start predicting."
        ),
    )
    stop = CLIOption(
        "--stop",
        default=None,
        type=int,
        help=(
            "For chunking, where to stop predicting."
        ),
    )

    train = CLICommand(
        "train",
        description="Train a model from configuration.",
        options=[
            input_name,
            # output_name,
            checkpoint_out.replace(required=True),
            CLIOption(
                "--validation",
                "-2",
                default=None,
                type=str,
                help="Validation dataset",
            ),
            config_opt,
            CLIOption(
                "--model",
                "-m",
                default=None,
                type=str,
                help=(
                    "Bundled model configuration alias "
                    "used to override box.model."
                ),
            ),
            pipeline_opt,
            set_opt,
            cache_opt,
        ],
        main=_train,
    )

    predict = CLICommand(
        "predict",
        description=(
            "Predict from a Box checkpoint."
        ),
        options=[
            input_name,
            start,
            stop,
            output_name,
            checkpoint_out,
            pipeline_opt,
            CLIOption(
                "--data",
                "-d",
                default=None,
                type=str,
            ),
            config_opt.replace(**{
                "required": False, 
                "default": None,
            }),
            set_opt,
            cache_opt,
        ],
        main=_predict,
    )

    app = CLIApp(
        app_name,
        version=__version__,
        description=(
            "Train, checkpoint, and predict using "
            "tree-based estimators."
        ),
        commands=[
            train,
            predict,
        ],
    )

    app.run()


if __name__ == "__main__":
    main()
