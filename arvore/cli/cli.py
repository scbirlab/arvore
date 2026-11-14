"""Command-line interface for arvore."""

from carabiner.cliutils import (
    CLIApp,
    CLICommand,
    CLIOption,
)

from .. import (
    __version__,
    app_name,
)
from .evaluate import _evaluate
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
    data_opt = CLIOption(
        "--data",
        "-d",
        default=None,
        type=str,
    )
    checkpoint_out = CLIOption(
        "--checkpoint",
        "-c",
        required=False,
        default=None,
        type=str,
        help="Where to save checkpoint",
    )
    allow_trees = CLIOption(
        "--trust-sklearn-trees",
        action="store_true",
        default=False,
        help=(
            "Allow loading sklearn Tree objects from "
            "trusted checkpoints."
        ),
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
            "dotted.path=value or a JSON/YAML file. ""May be repeated; later values take precedence."
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

    score_opt = CLIOption(
        "--score",
        nargs="*",
        default=None,
        type=str,
        help=(
            "Registered scorer or name(key=value) expression; "
            "optionally prefix output_name=. "
            "Quote expressions in the shell. "
            "May be combined with JSON-configured scorers."
        ),
    )

    skip_ref_save_opt = CLIOption(
        "--skip-ref-save",
        action="store_true",
        default=False,
        help=(
            "Do not prepare or checkpoint scorer reference resources. "
            "The fitted model and training data are still saved."
        ),
    )

    disaggregate_opt = CLIOption(
        "--disaggregate",
        action="store_true",
        help="Write individual ensemble-member predictions.",
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
            skip_ref_save_opt,
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
            allow_trees,
            start,
            stop,
            score_opt,
            disaggregate_opt,
            output_name,
            checkpoint_out,
            pipeline_opt,
            data_opt.replace(help="Data to predict on."),
            config_opt.replace(**{
                "required": False, 
                "default": None,
            }),
            set_opt,
            cache_opt,
        ],
        main=_predict,
    )

    evaluate = CLICommand(
        "evaluate",
        description="Evaluate a fitted Box on labelled data.",
        options=[
            input_name,
            allow_trees,
            data_opt.replace(help="Labelled evaluation dataset."),
            start,
            stop,
            CLIOption(
                "--metrics",
                nargs="*",
                default=None,
                help="Regression metrics to calculate.",
            ),
            CLIOption(
                "--result",
                "-r",
                required=False,
                default=None,
                type=str,
                help="Where to save metrics json",
            ),
            output_name,
            pipeline_opt,
            config_opt.replace(**{
                "required": False,
                "default": None,
            }),
            set_opt,
            cache_opt,
        ],
        main=_evaluate,
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
            evaluate,
        ],
    )

    app.run()


if __name__ == "__main__":
    main()
