from matplotlib import pyplot as plt

from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib.figure import Figure

from typing import Optional, Union, TypeAlias, Sequence
from .._cytonorm._cytonorm import CytoNorm

from ._utils import set_scatter_defaults, save_or_show, apply_vary_textures

NDArrayOfAxes: TypeAlias = "np.ndarray[Sequence[Sequence[Axes]], np.dtype[np.object_]]"


def emd(
    cnp: CytoNorm,
    colorby: str,
    data: Optional[pd.DataFrame] = None,
    channels: Optional[Union[list[str], str]] = None,
    labels: Optional[Union[list[str], str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    grid: Optional[str] = None,
    grid_n_cols: Optional[int] = None,
    vary_textures: bool = False,
    ax: Optional[Union[Axes, NDArrayOfAxes]] = None,
    return_fig: bool = False,
    show: bool = True,
    save: Optional[str] = None,
    **kwargs,
):
    """\
    EMD plot visualization.

    Parameters
    ----------
    colorby
        Selects the coloring of the data points. Can be any
        of 'label', 'channel' or 'improvement'.
        If 'improved', the data points are colored whether the
        EMD metric improved.
    data
        Optional. If not plotted from a cytonorm object, data
        can be passed. Has to contain the index columns,
        'label' and 'origin' (containing 'original' and
        'normalized').
    channels
        Optional. Can be used to select one or more channels.
    labels
        Optional. Can be used to select one or more cell labels.
    grid
        Whether to split the plots by the given variable. If
        left `None`, all data points are plotted into the same
        plot. Can be the same inputs as `colorby`.
    grid_n_cols
        The number of columns in the grid.
    vary_textures:
        If True, will plot different markers for the 'hue' variable.
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
    If `return_fig==True`, a :class:`~matplotlib.figure.Figure`.


    Examples
    --------
    .. plot::
        :context: close-figs

        import cytonormpy as cnp

        cn = cnp.example_cytonorm()
        cnp.pl.emd(cn,
                   colorby = "label",
                   s = 10,
                   linewidth = 0.4,
                   edgecolor = "black",
                   figsize = (4,4))
    """

    kwargs = set_scatter_defaults(kwargs)

    if data is None:
        emd_frame = cnp.emd_frame
    else:
        emd_frame = data

    df = _prepare_evaluation_frame(dataframe=emd_frame, channels=channels, labels=labels)
    df["improvement"] = (df["original"] - df["normalized"]) < 0
    df["improvement"] = df["improvement"].map({False: "improved", True: "worsened"})

    _check_grid_appropriate(df, grid)

    if grid is not None:
        fig, ax = _generate_scatter_grid(
            df=df,
            colorby=colorby,
            grid_by=grid,
            grid_n_cols=grid_n_cols,
            figsize=figsize,
            vary_textures=vary_textures,
            **kwargs,
        )
        ax_shape = ax.shape
        ax = ax.flatten()
        for i, _ in enumerate(ax):
            if not ax[i].axison:
                continue
            # we plot a line to compare the EMD values
            _draw_comp_line(ax[i])
            ax[i].set_title("EMD comparison")

        ax = ax.reshape(ax_shape)

    else:
        if ax is None:
            if figsize is None:
                figsize = (2, 2)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        else:
            fig = (None,)
            ax = ax
        assert ax is not None

        plot_kwargs = {"data": df, "x": "normalized", "y": "original", "hue": colorby, "ax": ax}
        if vary_textures:
            apply_vary_textures(plot_kwargs, df, colorby)
        assert isinstance(ax, Axes)
        sns.scatterplot(**plot_kwargs, **kwargs)
        _draw_comp_line(ax)
        ax.set_title("EMD comparison")
        if colorby is not None:
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")

    return save_or_show(ax=ax, fig=fig, save=save, show=show, return_fig=return_fig)


def mad(
    cnp: CytoNorm,
    colorby: str,
    data: Optional[pd.DataFrame] = None,
    file_name: Optional[Union[list[str], str]] = None,
    channels: Optional[Union[list[str], str]] = None,
    labels: Optional[Union[list[str], str]] = None,
    mad_cutoff: float = 0.25,
    grid: Optional[str] = None,
    grid_n_cols: Optional[int] = None,
    vary_textures: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[Union[Axes, NDArrayOfAxes]] = None,
    return_fig: bool = False,
    show: bool = True,
    save: Optional[str] = None,
    **kwargs,
):
    """\
    MAD plot visualization.

    Parameters
    ----------
    colorby
        Selects the coloring of the data points. Can be any
        of 'file_name', 'label', 'channel' or 'change'.
        If 'change', the data points are colored whether the
        MAD metric increased or decreased.
    data
        Optional. If not plotted from a cytonorm object, data
        can be passed. Has to contain the index columns 'file_name',
        'label' and 'origin' (containing 'original' and
        'normalized').
    file_name
        Optional. Can be used to select one or multiple files.
    channels
        Optional. Can be used to select one or more channels.
    labels
        Optional. Can be used to select one or more cell labels.
    mad_cutoff
        A red dashed line that is plotted, signifying a cutoff
    grid
        Whether to split the plots by the given variable. If
        left `None`, all data points are plotted into the same
        plot. Can be the same inputs as `colorby`.
    grid_n_cols
        The number of columns in the grid.
    vary_textures:
        If True, will plot different markers for the 'hue' variable.
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
        cn = cnp.example_cytonorm()
        cnp.pl.mad(cn,
                   colorby = "label",
                   s = 10,
                   linewidth = 0.4,
                   edgecolor = "black",
                   figsize = (4,4))
    """

    kwargs = set_scatter_defaults(kwargs)

    if data is None:
        mad_frame = cnp.mad_frame
    else:
        mad_frame = data

    df = _prepare_evaluation_frame(
        dataframe=mad_frame, file_name=file_name, channels=channels, labels=labels
    )
    df["change"] = (df["original"] - df["normalized"]) < 0
    df["change"] = df["change"].map({False: "decreased", True: "increased"})

    _check_grid_appropriate(df, grid)

    if grid is not None:
        fig, ax = _generate_scatter_grid(
            df=df,
            colorby=colorby,
            grid_by=grid,
            grid_n_cols=grid_n_cols,
            figsize=figsize,
            vary_textures=vary_textures,
            **kwargs,
        )
        ax_shape = ax.shape
        ax = ax.flatten()
        for i, _ in enumerate(ax):
            if not ax[i].axison:
                continue
            # we plot a line to compare the MAD values
            _draw_cutoff_line(ax[i], cutoff=mad_cutoff)
            ax[i].set_title("MAD comparison")

        ax = ax.reshape(ax_shape)

    else:
        if ax is None:
            if figsize is None:
                figsize = (2, 2)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        else:
            fig = (None,)
            ax = ax
        assert ax is not None

        plot_kwargs = {"data": df, "x": "normalized", "y": "original", "hue": colorby, "ax": ax}
        if vary_textures:
            apply_vary_textures(plot_kwargs, df, colorby)
        assert isinstance(ax, Axes)
        sns.scatterplot(**plot_kwargs, **kwargs)
        _draw_cutoff_line(ax, cutoff=mad_cutoff)
        ax.set_title("MAD comparison")
        if colorby is not None:
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")

    return save_or_show(ax=ax, fig=fig, save=save, show=show, return_fig=return_fig)


def _prepare_evaluation_frame(
    dataframe: pd.DataFrame,
    file_name: Optional[Union[list[str], str]] = None,
    channels: Optional[Union[list[str], str]] = None,
    labels: Optional[Union[list[str], str]] = None,
) -> pd.DataFrame:
    index_names = dataframe.index.names
    dataframe = dataframe.reset_index()
    melted = dataframe.melt(id_vars=index_names, var_name="channel", value_name="value")
    df = melted.pivot_table(
        index=[idx_name for idx_name in index_names if idx_name != "origin"] + ["channel"],
        columns="origin",
        values="value",
    ).reset_index()
    if file_name is not None:
        if not isinstance(file_name, list):
            file_name = [file_name]
        df = df.loc[df["file_name"].isin(file_name), :]

    if channels is not None:
        if not isinstance(channels, list):
            channels = [channels]
        df = df.loc[df["channel"].isin(channels), :]

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels]
        df = df.loc[df["label"].isin(labels), :]

    return df


def _unify_axes_dimensions(ax: Axes) -> None:
    axes_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    axes_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    axis_lims = (axes_min, axes_max)
    ax.set_xlim(axis_lims)
    ax.set_ylim(axis_lims)


def _draw_comp_line(ax: Axes) -> None:
    _unify_axes_dimensions(ax)

    comp_line_x = list(ax.get_xlim())
    comp_line_y = comp_line_x
    ax.plot(comp_line_x, comp_line_y, color="red", linestyle="--")
    ax.set_xlim(comp_line_x[0], comp_line_x[1])
    ax.set_ylim(comp_line_x[0], comp_line_x[1])
    return


def _draw_cutoff_line(ax: Axes, cutoff: float) -> None:
    _unify_axes_dimensions(ax)

    upper_bound_x = list(ax.get_xlim())
    upper_bound_y = [val + cutoff for val in upper_bound_x]
    lower_bound_x = list(ax.get_ylim())
    lower_bound_y = [val - cutoff for val in lower_bound_x]
    ax.plot(upper_bound_x, upper_bound_y, color="red", linestyle="--")
    ax.plot(upper_bound_x, lower_bound_y, color="red", linestyle="--")
    ax.set_xlim(upper_bound_x[0], upper_bound_x[1])
    ax.set_ylim(upper_bound_x[0], upper_bound_x[1])


def _check_grid_appropriate(df: pd.DataFrame, grid_by: Optional[str]) -> None:
    if grid_by is not None:
        if df[grid_by].nunique() == 1:
            error_msg = "Only one unique value for the grid variable. "
            error_msg += "A Grid is not possible."
            raise ValueError(error_msg)
    return


def _generate_scatter_grid(
    df: pd.DataFrame,
    grid_by: str,
    grid_n_cols: Optional[int],
    figsize: tuple[float, float],
    colorby: Optional[str],
    vary_textures: bool,
    **scatter_kwargs: Optional[dict],
) -> tuple[Figure, NDArrayOfAxes]:
    n_cols, n_rows, figsize = _get_grid_sizes(
        df=df, grid_by=grid_by, grid_n_cols=grid_n_cols, figsize=figsize
    )

    # calculate it to remove empty axes later
    total_plots = n_cols * n_rows

    hue = None if colorby == grid_by else colorby
    plot_params = {"x": "normalized", "y": "original", "hue": hue}

    if vary_textures:
        apply_vary_textures(plot_params, df, colorby)

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize, sharex=True, sharey=True)
    ax = ax.flatten()
    i = 0

    for i, grid_param in enumerate(df[grid_by].unique()):
        sns.scatterplot(
            data=df[df[grid_by] == grid_param], **plot_params, **scatter_kwargs, ax=ax[i]
        )
        ax[i].set_title(grid_param)
        if hue is not None:
            handles, labels = ax[i].get_legend_handles_labels()
            ax[i].legend_.remove()

    if i < total_plots:
        for j in range(total_plots):
            if j > i:
                ax[j].axis("off")

    ax = ax.reshape(n_cols, n_rows)

    if hue is not None:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc="center left", title=colorby)

    return fig, ax


def _get_grid_sizes(
    df: pd.DataFrame,
    grid_by: str,
    grid_n_cols: Optional[int],
    figsize: Optional[tuple[float, float]],
) -> tuple:
    n_plots = df[grid_by].nunique()
    if grid_n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_plots)))
    else:
        n_cols = grid_n_cols

    n_rows = int(np.ceil(n_plots / n_cols))

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    return n_cols, n_rows, figsize
