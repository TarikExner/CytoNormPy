from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd

from matplotlib.figure import Figure

from typing import Optional, Literal, Union, cast

from .._cytonorm import CytoNorm

from ._utils import (
    set_scatter_defaults,
    modify_axes,
    modify_legend,
    save_or_show,
    apply_vary_textures,
)


def scatter(
    cnp: CytoNorm,
    file_name: str,
    x_channel: str,
    y_channel: str,
    x_scale: Literal["biex", "log", "linear"] = "linear",
    y_scale: Literal["biex", "log", "linear"] = "linear",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    legend_labels: Optional[list[str]] = None,
    subsample: Optional[int] = None,
    linthresh: float = 500,
    display_reference: bool = True,
    vary_textures: bool = False,
    figsize: tuple[float, float] = (2, 2),
    ax: Optional[Axes] = None,
    return_fig: bool = False,
    show: bool = True,
    save: Optional[str] = None,
    **kwargs,
) -> Optional[Union[Figure, Axes]]:
    """\
    Scatterplot visualization.

    Parameters
    ----------
    file_name
        The file name of the file that is supposed
        to be plotted.
    x_channel
        The channel plotted on the x-axis.
    y_channel
        The channel plotted on the y-axis.
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
    legend_labels
        The labels displayed in the legend.
    subsample
        A number of events to subsample to. Can prevent
        overcrowding of the plot.
    linthresh
        The value to switch from a linear to a log axis.
        Ignored if neither x- nor y-scale are `biex`.
    display_reference
        Whether to display the reference data from
        that batch as well. Defaults to True.
    vary_textures
        If True, use different marker shapes for each 'origin' category
        by passing `style="origin"` and a `markers` mapping to seaborn.
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
        cnp.pl.scatter(cn,
                       cn._datahandler.metadata.validation_file_names[0],
                       x_channel = "Ho165Di",
                       y_channel = "Yb172Di",
                       x_scale = "linear",
                       y_scale = "linear",
                       figsize = (4,4),
                       s = 10,
                       linewidth = 0.4,
                       edgecolor = "black")
    .. note::
        If you want additional separation of the individual point classes,
        you can pass 'vary_textures=True'.

    .. plot::
        :context: close-figs

        import cytonormpy as cnp

        cn = cnp.example_cytonorm()
        cnp.pl.scatter(cn,
                       cn._datahandler.metadata.validation_file_names[0],
                       x_channel = "Ho165Di",
                       y_channel = "Yb172Di",
                       x_scale = "linear",
                       y_scale = "linear",
                       vary_textures = True,
                       figsize = (4,4),
                       s = 10,
                       linewidth = 0.4,
                       edgecolor = "black")

    """

    data = _prepare_data(cnp, file_name, display_reference, channels=None, subsample=subsample)

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    else:
        fig = ax.figure
        ax = ax
    assert ax is not None

    hues = data.index.get_level_values("origin").unique().sort_values()
    plot_kwargs = {
        "data": data,
        "hue": "origin",
        "hue_order": hues,
        "x": x_channel,
        "y": y_channel,
        "ax": ax,
    }

    if vary_textures:
        apply_vary_textures(plot_kwargs, data.reset_index(), "origin")

    kwargs = set_scatter_defaults(kwargs)

    sns.scatterplot(**plot_kwargs, **kwargs)

    modify_axes(ax=ax, x_scale=x_scale, y_scale=y_scale, xlim=xlim, ylim=ylim, linthresh=linthresh)

    modify_legend(ax=ax, legend_labels=legend_labels)

    return save_or_show(ax=ax, fig=fig, save=save, show=show, return_fig=return_fig)


def _prepare_data(
    cnp: CytoNorm,
    file_name: str,
    display_reference: bool,
    channels: Optional[Union[list[str], str]],
    subsample: Optional[int],
) -> pd.DataFrame:
    original_df = cnp._datahandler.get_dataframe(file_name)

    normalized_df = cnp._normalize_file(
        df=original_df.copy(), batch=cnp._datahandler.metadata.get_batch(file_name)
    )

    if display_reference is True:
        ref_df = cnp._datahandler.get_corresponding_ref_dataframe(file_name)
        ref_df["origin"] = "reference"
        ref_df = ref_df.set_index("origin", append=True, drop=True)
        ref_df = _select_index_levels(ref_df)
    else:
        ref_df = None

    original_df["origin"] = "original"
    normalized_df["origin"] = "transformed"

    original_df = original_df.set_index("origin", append=True, drop=True)
    normalized_df = normalized_df.set_index("origin", append=True, drop=True)

    original_df = _select_index_levels(original_df)
    normalized_df = _select_index_levels(normalized_df)

    # we clean up the indices in order to not mess up the

    if ref_df is not None:
        data = pd.concat([normalized_df, original_df, ref_df], axis=0)
    else:
        data = pd.concat([normalized_df, original_df], axis=0)

    if channels is not None:
        data = data[channels]

    if subsample:
        data = data.sample(n=subsample)
    else:
        data = data.sample(frac=1)  # overlays are better shuffled

    return cast(pd.DataFrame, data)


def _select_index_levels(df: pd.DataFrame):
    index_levels_to_keep = ["origin", "reference", "batch", "file_name"]
    for name in df.index.names:
        if name not in index_levels_to_keep:
            df = df.droplevel(name)
    return df
