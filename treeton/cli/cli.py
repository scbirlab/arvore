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
        required=True,
        type=str,
    )

    config_opt = CLIOption(
        "--config",
        "-c",
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

    train = CLICommand(
        "train",
        description="Train a model from configuration.",
        options=[
            input_name,
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
            CLIOption(
                "--output",
                "-o",
                required=True,
                type=str,
                help="Where to save model",
            ),
            set_opt,
            CLIOption(
                "--cache",
                default=None,
                type=str,
                help="Cache directory for data loading and preprocessing.",
            ),
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
            pipeline_opt,
            CLIOption(
                "--data",
                "-d",
                required=True,
                type=str,
            ),
            config_opt.replace(**{
                "required": False, 
                "default": None,
            }),
            set_opt,
            CLIOption(
                "--output",
                "-o",
                required=True,
                type=str,
            ),
            CLIOption(
                "--cache",
                default=None,
                type=str,
            ),
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
