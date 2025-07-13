from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from typing import Optional, Literal, Union, TypeAlias, Sequence
from .._cytonorm._cytonorm import CytoNorm

from ._utils import modify_axes, save_or_show, DASH_STYLES
from ._scatter import _prepare_data

NDArrayOfAxes: TypeAlias = "np.ndarray[Sequence[Sequence[Axes]], np.dtype[np.object_]]"


def histogram(
    cnp: CytoNorm,
    file_name: str,
    x_channel: Optional[str] = None,
    x_scale: Literal["biex", "log", "linear"] = "linear",
    y_scale: Literal["biex", "log", "linear"] = "linear",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    linthresh: float = 500,
    subsample: Optional[int] = None,
    display_reference: bool = True,
    grid: Optional[Literal["channels"]] = None,
    grid_n_cols: Optional[int] = None,
    channels: Optional[Union[list[str], str]] = None,
    vary_textures: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[Union[NDArrayOfAxes, Axes]] = None,
    return_fig: bool = False,
    show: bool = True,
    save: Optional[str] = None,
    **kwargs,
) -> Optional[Union[Figure, Axes]]:
    """\
    Histogram visualization.

    Parameters
    ----------
    file_name
        The file name of the file that is supposed
        to be plotted.
    x_channel
        The channel plotted on the x-axis.
    x_scale
        The scale type of the x-axis. Can be one
        of `biex`, `linear` or `log`. Defaults to
        `biex`.
    y_scale
        The scale type of the y-axis. Can be one
        of `biex`, `linear` or `log`. Defaults to
        `biex`.
    legend_labels
        The labels displayed in the legend.
    linthresh
        The value to switch from a linear to a log axis.
        Ignored if neither x- nor y-scale are `biex`.
    subsample
        A number of events to subsample to. Can prevent
        overcrowding of the plot.
    display_reference
        Whether to display the reference data from
        that batch as well. Defaults to True.
    grid
        Can be'channels'. Will plot a grid where each
        channel gets its own plot. A `file_name` has to be
        provided.
    channels
        Optional. Can be used to select one or more channels
        that will be plotted in the grid.
    vary_textures
        If True, apply different line styles per `origin` category.
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
        sns.scatterplot.

    Returns
    -------
    If `show==False`, a :class:`~matplotlib.axes.Axes`.


    Examples
    --------
    .. plot::
        :context: close-figs

        import cytonormpy as cnp

        cn = cnp.example_cytonorm()
        cnp.pl.histogram(cn,
                         cn._datahandler.metadata.validation_file_names[0],
                         x_channel = "Ho165Di",
                         x_scale = "linear",
                         y_scale = "linear",
                         figsize = (4,4))

    .. note::
        If you want additional separation of the individual point classes,
        you can pass 'vary_textures=True'.

    .. plot::
        :context: close-figs

        import cytonormpy as cnp

        cn = cnp.example_cytonorm()
        cnp.pl.histogram(cn,
                         cn._datahandler.metadata.validation_file_names[0],
                         x_channel = "Ho165Di",
                         x_scale = "linear",
                         y_scale = "linear",
                         figsize = (4,4),
                         vary_textures = True)

    """
    if x_channel is None and grid is None:
        raise ValueError("Either provide a gate or set 'grid' to 'channels'")
    if grid == "file_name":
        raise NotImplementedError("Currently not supported")
        # raise ValueError("A Grid by file_name needs a x_channel")
    if grid == "channels" and file_name is None:
        raise ValueError("A Grid by channels needs a file_name")

    data = _prepare_data(cnp, file_name, display_reference, channels, subsample=subsample)

    hues = data.index.get_level_values("origin").unique().sort_values()

    dash_styles = DASH_STYLES
    style_map = {origin: dash_styles[i % len(dash_styles)] for i, origin in enumerate(hues)}

    kde_kwargs = {}

    if grid is not None:
        assert grid == "channels"
        n_cols, n_rows, figsize = _get_grid_sizes_channels(
            df=data, grid_n_cols=grid_n_cols, figsize=figsize
        )

        # calculate it to remove empty axes later
        total_plots = n_cols * n_rows

        ax: NDArrayOfAxes
        fig, ax = plt.subplots(
            ncols=n_cols, nrows=n_rows, figsize=figsize, sharex=False, sharey=False
        )
        ax = ax.flatten()
        i = 0

        assert ax is not None

        for i, grid_param in enumerate(data.columns):
            plot_kwargs = {
                "data": data,
                "hue": "origin",
                "hue_order": hues,
                "x": grid_param,
                "ax": ax[i],
            }
            ax[i] = sns.kdeplot(**plot_kwargs, **kde_kwargs, **kwargs)

            if vary_textures:
                _apply_textures_and_legend(ax[i], hues, style_map)

            modify_axes(
                ax=ax[i],
                x_scale=x_scale,
                y_scale=y_scale,
                xlim=xlim,
                ylim=ylim,
                linthresh=linthresh,
            )
            legend = ax[i].legend_
            handles = legend.legend_handles
            labels = [t.get_text() for t in legend.get_texts()]

            ax[i].legend_.remove()
            ax[i].set_title(grid_param)
        if i < total_plots:
            for j in range(total_plots):
                if j > i:
                    ax[j].axis("off")

        ax = ax.reshape(n_cols, n_rows)

        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc="center left", title="origin")

    else:
        plot_kwargs = {
            "data": data,
            "hue": "origin",
            "hue_order": hues,
            "x": x_channel,
            "ax": ax,
        }
        if ax is None:
            if figsize is None:
                figsize = (2, 2)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        else:
            fig = (None,)
            ax = ax
        assert ax is not None

        ax = sns.kdeplot(**plot_kwargs, **kde_kwargs, **kwargs)

        if vary_textures:
            _apply_textures_and_legend(ax, hues, style_map)

        sns.move_legend(ax, bbox_to_anchor=(1.01, 0.5), loc="center left")

        modify_axes(
            ax=ax, x_scale=x_scale, y_scale=y_scale, xlim=xlim, ylim=ylim, linthresh=linthresh
        )

    return save_or_show(ax=ax, fig=fig, save=save, show=show, return_fig=return_fig)


def _get_grid_sizes_channels(
    df: pd.DataFrame, grid_n_cols: Optional[int], figsize: Optional[tuple[float, float]]
) -> tuple:
    n_plots = len(df.columns)
    if grid_n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_plots)))
    else:
        n_cols = grid_n_cols

    n_rows = int(np.ceil(n_plots / n_cols))

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    return n_cols, n_rows, figsize


def _apply_textures_and_legend(ax: Axes, hues: list[str], style_map: dict[str, str]) -> None:
    """
    1) Apply the linestyle from style_map to each line in ax.lines,
       assuming they come out in the same order as hues.
    2) Remove any existing legend and draw a new one with correct labels.
    """
    for idx, line in enumerate(ax.lines):
        origin = hues[idx]
        line.set_linestyle(style_map[origin])

    colors = [line.get_color() for line in ax.lines[: len(hues)]]
    handles = [
        Line2D([], [], color=colors[i], linestyle=style_map[origin], label=origin)
        for i, origin in enumerate(hues)
    ]

    if ax.legend_:
        ax.legend_.remove()
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 0.5), loc="center left", title="origin")
