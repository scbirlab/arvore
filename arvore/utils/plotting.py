"""Plotting utilities."""

from carabiner.mpl import add_legend, grid, figsaver
import pandas as pd
import numpy as np


def _plot_history(
    lightning_csv, 
    filename: str
) -> None:

    data_to_plot = (
        pd.read_csv(lightning_csv)
        .groupby(['epoch', 'step'])
        .agg("mean")
        .reset_index()
    )

    fig, ax = grid(aspect_ratio=1.5)
    for _y in ('val_loss', 'loss', 'learning_rate'):
        if _y in data_to_plot:
            ax.plot(
                'step', _y, 
                data=data_to_plot, 
                label=_y,
            )
            ax.scatter(
                'step', _y, 
                data=data_to_plot,
                s=1.,
            )
    add_legend(ax)
    ax.set(
        xlabel='Training step', 
        ylabel='Loss', 
        yscale='log',
    )
    figsaver(format="png")(fig, name=filename, df=data_to_plot)
    return None


def _plot_prediction_scatter(
    df,
    filename: str,
    x: str = "__prediction__",
    y: str = "labels"
) -> None:
    fig, ax = grid()
    ax.scatter(
        x, y,
        data=df,
        s=1.,
    )
    ax.plot(
        ax.get_ylim(),
        ax.get_ylim(),
        color='dimgrey',
        zorder=-5,
    )
    ax.set(
        xlabel=f"Predicted ({x})", 
        ylabel=f"Observed ({y})",
    )
    figsaver(format="png")(
        fig,
        filename,
        df=df,
    )
    return None
