from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib.figure import Figure

from typing import Optional, Literal, Union
from .._cytonorm._cytonorm import CytoNorm

from ._utils import modify_axes, save_or_show


def splineplot(
    cnp: CytoNorm,
    file_name: str,
    channel: str,
    label_quantiles: Optional[list[float]] = [0.1, 0.25, 0.5, 0.75, 0.9],
    x_scale: Literal["biex", "log", "linear"] = "linear",
    y_scale: Literal["biex", "log", "linear"] = "linear",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    linthresh: float = 500,
    figsize: tuple[float, float] = (2, 2),
    ax: Optional[Axes] = None,
    return_fig: bool = False,
    show: bool = True,
    save: Optional[str] = None,
    **kwargs,
) -> Optional[Union[Figure, Axes]]:
    """\
    Splineplot visualization.

    Parameters
    ----------
    file_name
        The file name of the file that is supposed
        to be plotted.
    channel
        The channel to be plotted.
    label_quantiles
        A list of the quantiles that are labeled in the plot.
    x_scale
        The scale type of the x-axis. Can be one
        of `biex`, `linear` or `log`. Defaults to
        `biex`.
    y_scale
        The scale type of the y-axis. Can be one
        of `biex`, `linear` or `log`. Defaults to
        `biex`.
    xlim
        Sets the x-axis limits.
    ylim
        Sets the y-axis limits.
    linthresh
        The value to switch from a linear to a log axis.
        Ignored if neither x- nor y-scale are `biex`.
    ax
        A Matplotlib Axes to plot into.
    return_fig
        Returns the figure. Defaults to False.
    show
        Whether to show the figure.
    save
        A string specifying a file path. Defaults
        to None, where no image is saved.
    kwargs
        keyword arguments ultimately passed to
        sns.lineplot.

    Returns
    -------
    If `show==False`, a :class:`~matplotlib.axes.Axes`.

    Examples
    --------
    .. plot::
        :context: close-figs

        import cytonormpy as cnp

        cn = cnp.example_cytonorm()
        cnp.pl.splineplot(cn,
                          cn._datahandler.metadata.validation_file_names[0],
                          channel = "Tb159Di",
                          x_scale = "linear",
                          y_scale = "linear",
                          figsize = (4,4))

    """

    if label_quantiles is None:
        label_quantiles = []

    expr_quantiles = cnp._expr_quantiles
    quantiles: np.ndarray = expr_quantiles.quantiles

    batches = cnp.batches
    channels = cnp.channels
    batch_idx = batches.index(cnp._datahandler.metadata.get_batch(file_name))
    ch_idx = channels.index(channel)
    channel_quantiles = np.nanmean(
        expr_quantiles.get_quantiles(
            channel_idx=ch_idx,
            batch_idx=batch_idx,
            cluster_idx=None,
            quantile_idx=None,
            flattened=False,
        ),
        axis=expr_quantiles._cluster_axis,
    )

    goal_quantiles = np.nanmean(
        cnp._goal_distrib.get_quantiles(
            channel_idx=ch_idx,
            batch_idx=None,
            cluster_idx=None,
            quantile_idx=None,
            flattened=False,
        ),
        axis=expr_quantiles._cluster_axis,
    )
    df = pd.DataFrame(
        data={"original": channel_quantiles.flatten(), "goal": goal_quantiles.flatten()},
        index=quantiles.flatten(),
    )

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    else:
        fig = (None,)
        ax = ax
    assert ax is not None

    sns.lineplot(data=df, x="original", y="goal", ax=ax, **kwargs)
    ax.set_title(channel)
    modify_axes(ax=ax, x_scale=x_scale, y_scale=y_scale, xlim=xlim, ylim=ylim, linthresh=linthresh)

    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    xmin, xmax = ax.get_xlim()
    for q in label_quantiles:
        plt.vlines(
            x=df.loc[df.index == q, "original"].iloc[0],
            ymin=ylims[0],
            ymax=df.loc[df.index == q, "goal"].iloc[0],
            color="black",
            linewidth=0.4,
        )
        plt.hlines(
            y=df.loc[df.index == q, "goal"].iloc[0],
            xmin=xlims[0],
            xmax=df.loc[df.index == q, "original"].iloc[0],
            color="black",
            linewidth=0.4,
        )
        plt.text(
            x=xmin + 0.01 * (xmax - xmin),
            y=df.loc[df.index == q, "goal"].iloc[0] + ((ylims[1] - ylims[0]) / 200),
            s=f"Q{int(q * 100)}",
        )

    return save_or_show(ax=ax, fig=fig, save=save, show=show, return_fig=return_fig)
