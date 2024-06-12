from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib.figure import Figure

from typing import Optional, Literal, Union
from .._cytonorm._cytonorm import CytoNorm


class Plotter:
    """\
    Allows plotting from the cytonorm object.
    Implements scatter plot and histogram for
    the channels, and a splinefunc plot to
    visualize the splines.
    """

    def __init__(self,
                 cytonorm: CytoNorm):
        self.cnp = cytonorm

    def _prepare_data(self,
                      file_name: str,
                      display_reference: bool) -> pd.DataFrame:

        original_df = self.cnp._datahandler \
            .get_dataframe(file_name)

        normalized_df = self.cnp.\
            _transform_file(
                df = original_df.copy(),
                batch = self.cnp._datahandler.get_batch(file_name)
            )

        if display_reference is True:
            ref_df = self.cnp._datahandler \
                .get_corresponding_ref_dataframe(file_name)
            ref_df["origin"] = "reference"
        else:
            ref_df = None

        original_df["origin"] = "original"
        normalized_df["origin"] = "transformed"

        if ref_df is not None:
            data = pd.concat([normalized_df,
                              original_df,
                              ref_df], axis = 0)
        else:
            data = pd.concat([normalized_df,
                              original_df], axis = 0)

        data = data.sample(frac = 1)  # overlays are better shuffled

        return data

    def _handle_axis(self,
                     ax: Axes,
                     x_scale: str,
                     y_scale: str,
                     linthresh: Optional[float],
                     xlim: Optional[tuple[float, float]],
                     ylim: Optional[tuple[float, float]]) -> None:

        # Axis scale
        x_scale_kwargs: dict[str, Optional[Union[float, str]]] = {
            "value": x_scale if x_scale != "biex" else "symlog"
        }
        y_scale_kwargs: dict[str, Optional[Union[float, str]]] = {
            "value": y_scale if y_scale != "biex" else "symlog"
        }

        if x_scale == "biex":
            x_scale_kwargs["linthresh"] = linthresh
        if y_scale == "biex":
            y_scale_kwargs["linthresh"] = linthresh

        ax.set_xscale(**x_scale_kwargs)
        ax.set_yscale(**y_scale_kwargs)

        # Axis limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        return

    def _handle_legend(self,
                       ax: Axes,
                       legend_labels: Optional[list[str]]) -> None:
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if legend_labels:
            labels = legend_labels
        ax.legend(
            handles, labels,
            loc = "center left",
            bbox_to_anchor = (1.01, 0.5)
        )
        return

    def _scatter_defaults(self,
                          kwargs: dict) -> dict:
        kwargs["s"] = kwargs.get("s", 2)
        kwargs["edgecolor"] = kwargs.get("edgecolor", "black")
        kwargs["linewidth"] = kwargs.get("linewidth", 0.1)
        return kwargs

    def histogram(self,
                  file_name: str,
                  x_channel: str,
                  x_scale: Literal["biex", "log", "linear"] = "biex",
                  y_scale: Literal["biex", "log", "linear"] = "linear",
                  xlim: Optional[tuple[float, float]] = None,
                  ylim: Optional[tuple[float, float]] = None,
                  linthresh: float = 500,
                  display_reference: bool = True,
                  figsize: tuple[float, float] = (2, 2),
                  ax: Optional[Axes] = None,
                  return_fig: bool = False,
                  show: bool = True,
                  save: Optional[str] = None,
                  **kwargs) -> Optional[Union[Figure, Axes]]:
        """\
        Histogram visualization.

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
        legend_labels
            The labels displayed in the legend.
        linthresh
            The value to switch from a linear to a log axis.
            Ignored if neither x- nor y-scale are `biex`.
        display_reference
            Whether to display the reference data from
            that batch as well. Defaults to True.
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
            cnpl = cnp.Plotter(cytonorm = cn)

            cnpl.histogram(cn._datahandler.validation_file_names[0],
                           x_channel = "Nd142Di",
                           x_scale = "linear",
                           y_scale = "linear",
                           figsize = (4,4))

        """

        data = self._prepare_data(file_name,
                                  display_reference)

        if ax is None:
            fig, ax = plt.subplots(ncols = 1,
                                   nrows = 1,
                                   figsize = figsize)
        else:
            fig = None,
            ax = ax
        assert ax is not None
        plot_kwargs = {
            "data": data,
            "hue": "origin",
            "x": x_channel,
            "ax": ax
        }
        kde_kwargs = {}
        ax = sns.kdeplot(**plot_kwargs,
                         **kde_kwargs,
                         **kwargs)

        sns.move_legend(ax,
                        bbox_to_anchor = (1.01, 0.5),
                        loc = "center left")

        self._handle_axis(ax = ax,
                          x_scale = x_scale,
                          y_scale = y_scale,
                          xlim = xlim,
                          ylim = ylim,
                          linthresh = linthresh)

        return self._save_or_show(ax = ax,
                                  fig = fig,
                                  save = save,
                                  show = show,
                                  return_fig = return_fig)

    def scatter(self,
                file_name: str,
                x_channel: str,
                y_channel: str,
                x_scale: Literal["biex", "log", "linear"] = "biex",
                y_scale: Literal["biex", "log", "linear"] = "biex",
                xlim: Optional[tuple[float, float]] = None,
                ylim: Optional[tuple[float, float]] = None,
                legend_labels: Optional[list[str]] = None,
                linthresh: float = 500,
                display_reference: bool = True,
                figsize: tuple[float, float] = (2, 2),
                ax: Optional[Axes] = None,
                return_fig: bool = False,
                show: bool = True,
                save: Optional[str] = None,
                **kwargs) -> Optional[Union[Figure, Axes]]:
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
        linthresh
            The value to switch from a linear to a log axis.
            Ignored if neither x- nor y-scale are `biex`.
        display_reference
            Whether to display the reference data from
            that batch as well. Defaults to True.
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
            cnpl = cnp.Plotter(cytonorm = cn)

            cnpl.scatter(cn._datahandler.validation_file_names[0],
                         x_channel = "Nd144Di",
                         y_channel = "Sm147Di",
                         x_scale = "linear",
                         y_scale = "linear",
                         figsize = (4,4),
                         s = 10,
                         linewidth = 0.4,
                         edgecolor = "black")


        """

        data = self._prepare_data(file_name,
                                  display_reference)

        if ax is None:
            fig, ax = plt.subplots(ncols = 1,
                                   nrows = 1,
                                   figsize = figsize)
        else:
            fig = None,
            ax = ax
        assert ax is not None

        plot_kwargs = {
            "data": data,
            "hue": "origin",
            "x": x_channel,
            "y": y_channel,
            "ax": ax
        }

        kwargs = self._scatter_defaults(kwargs)

        sns.scatterplot(**plot_kwargs,
                        **kwargs)

        self._handle_axis(ax = ax,
                          x_scale = x_scale,
                          y_scale = y_scale,
                          xlim = xlim,
                          ylim = ylim,
                          linthresh = linthresh)

        self._handle_legend(ax = ax,
                            legend_labels = legend_labels)

        return self._save_or_show(ax = ax,
                                  fig = fig,
                                  save = save,
                                  show = show,
                                  return_fig = return_fig)

    def splineplot(self,
                   file_name: str,
                   channel: str,
                   label_quantiles: Optional[list[float]] = [0.1, 0.25, 0.5, 0.75, 0.9],  # noqa
                   x_scale: Literal["biex", "log", "linear"] = "biex",
                   y_scale: Literal["biex", "log", "linear"] = "biex",
                   xlim: Optional[tuple[float, float]] = None,
                   ylim: Optional[tuple[float, float]] = None,
                   linthresh: float = 500,
                   figsize: tuple[float, float] = (2, 2),
                   ax: Optional[Axes] = None,
                   return_fig: bool = False,
                   show: bool = True,
                   save: Optional[str] = None,
                   **kwargs) -> Optional[Union[Figure, Axes]]:
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
            cnpl = cnp.Plotter(cytonorm = cn)

            cnpl.splineplot(cn._datahandler.validation_file_names[0],
                            channel = "Nd144Di",
                            x_scale = "linear",
                            y_scale = "linear",
                            figsize = (4,4))

        """

        if label_quantiles is None:
            label_quantiles = []

        expr_quantiles = self.cnp._expr_quantiles
        quantiles: np.ndarray = expr_quantiles.quantiles

        batches = self.cnp.batches
        channels = self.cnp.channels
        batch_idx = batches.index(self.cnp._datahandler.get_batch(file_name))
        ch_idx = channels.index(channel)
        channel_quantiles = np.nanmean(
            expr_quantiles.get_quantiles(
                channel_idx = ch_idx,
                batch_idx = batch_idx,
                cluster_idx = None,
                quantile_idx = None,
                flattened = False),
            axis = expr_quantiles._cluster_axis
        )

        goal_quantiles = np.nanmean(
            self.cnp._goal_distrib.get_quantiles(
                channel_idx = ch_idx,
                batch_idx = None,
                cluster_idx = None,
                quantile_idx = None,
                flattened = False),
            axis = expr_quantiles._cluster_axis
        )
        df = pd.DataFrame(
            data = {
                "original": channel_quantiles.flatten(),
                "goal": goal_quantiles.flatten()
            },
            index = quantiles.flatten()
        )

        if ax is None:
            fig, ax = plt.subplots(ncols = 1,
                                   nrows = 1,
                                   figsize = figsize)
        else:
            fig = None,
            ax = ax
        assert ax is not None

        sns.lineplot(
            data = df,
            x = "original",
            y = "goal",
            ax = ax,
            **kwargs
        )
        self._handle_axis(ax = ax,
                          x_scale = x_scale,
                          y_scale = y_scale,
                          xlim = xlim,
                          ylim = ylim,
                          linthresh = linthresh)

        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        for q in label_quantiles:
            plt.vlines(x = df.loc[df.index == q, "original"].iloc[0],
                       ymin = ylims[0],
                       ymax = df.loc[df.index == q, "goal"].iloc[0],
                       color = "black",
                       linewidth = 0.4)
            plt.hlines(y = df.loc[df.index == q, "goal"].iloc[0],
                       xmin = xlims[0],
                       xmax = df.loc[df.index == q, "original"].iloc[0],
                       color = "black",
                       linewidth = 0.4)
            plt.text(x = -0.7,
                     y = df.loc[df.index == q, "goal"].iloc[0] + ((ylims[1] - ylims[0]) / 200),
                     s = f"Q{int(q*100)}")

        return self._save_or_show(ax = ax,
                                  fig = fig,
                                  save = save,
                                  show = show,
                                  return_fig = return_fig)

    def _save_or_show(self,
                      ax: Axes,
                      fig: Optional[Figure],
                      save: Optional[str],
                      show: bool,
                      return_fig: bool) -> Optional[Union[Figure, Axes]]:

        if save:
            plt.savefig(save, dpi = 300, bbox_inches = "tight")

        if show:
            plt.show()

        if return_fig:
            return fig

        return ax if not show else None
