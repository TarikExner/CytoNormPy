from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib.figure import Figure

from typing import Optional, Literal, Union, TypeAlias, Sequence
from .._cytonorm._cytonorm import CytoNorm

NDArrayOfAxes: TypeAlias = 'np.ndarray[Sequence[Sequence[Axes]], np.dtype[np.object_]]'

class Plotter:
    """\
    Allows plotting from the cytonorm object.
    Implements scatter plot and histogram for
    the channels, and a splinefunc plot to
    visualize the splines. Further, EMD and MAD plots
    are implemented in order to visualize the
    evaluation metrics.
    """

    def __init__(self,
                 cytonorm: CytoNorm):
        self.cnp = cytonorm

    def emd(self,
            colorby: str,
            data: Optional[pd.DataFrame] = None,
            channels: Optional[Union[list[str], str]] = None,
            labels: Optional[Union[list[str], str]] = None,
            figsize: Optional[tuple[float, float]] = None,
            grid: Optional[str] = None,
            grid_n_cols: Optional[int] = None,
            ax: Optional[Union[Axes, NDArrayOfAxes]] = None,
            return_fig: bool = False,
            show: bool = True,
            save: Optional[str] = None,
            **kwargs):
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
            cnpl = cnp.Plotter(cytonorm = cn)

            cnpl.emd(colorby = "label",
                     s = 10,
                     linewidth = 0.4,
                     edgecolor = "black",
                     figsize = (4,4))
        """

        kwargs = self._scatter_defaults(kwargs)

        if data is None:
            emd_frame = self.cnp.emd_frame
        else:
            emd_frame = data

        df = self._prepare_evaluation_frame(dataframe = emd_frame,
                                            channels = channels,
                                            labels = labels)
        df["improvement"] = (df["original"] - df["normalized"]) < 0
        df["improvement"] = df["improvement"].map(
            {False: "improved", True: "worsened"}
        )

        self._check_grid_appropriate(df, grid)

        if grid is not None:
            fig, ax = self._generate_scatter_grid(
                df = df,
                colorby = colorby,
                grid_by = grid,
                grid_n_cols = grid_n_cols,
                figsize = figsize,
                **kwargs
            )
            ax_shape = ax.shape
            ax = ax.flatten()
            for i, _ in enumerate(ax):
                if not ax[i].axison:
                    continue
                # we plot a line to compare the EMD values
                self._draw_comp_line(ax[i])
                ax[i].set_title("EMD comparison")

            ax = ax.reshape(ax_shape)

        else:
            if ax is None:
                if figsize is None:
                    figsize = (2,2)
                fig, ax = plt.subplots(ncols = 1,
                                       nrows = 1,
                                       figsize = figsize)
            else:
                fig = None,
                ax = ax
            assert ax is not None

            plot_kwargs = {
                "data": df,
                "x": "normalized",
                "y": "original",
                "hue": colorby,
                "ax": ax
            }
            assert isinstance(ax, Axes)
            sns.scatterplot(**plot_kwargs,
                            **kwargs)
            self._draw_comp_line(ax)
            ax.set_title("EMD comparison")
            if colorby is not None:
                ax.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left")

        return self._save_or_show(
            ax = ax,
            fig = fig,
            save = save,
            show = show,
            return_fig = return_fig
        )

    def mad(self,
            colorby: str,
            data: Optional[pd.DataFrame] = None,
            file_name: Optional[Union[list[str], str]] = None,
            channels: Optional[Union[list[str], str]] = None,
            labels: Optional[Union[list[str], str]] = None,
            mad_cutoff: float = 0.25,
            grid: Optional[str] = None,
            grid_n_cols: Optional[int] = None,
            figsize: Optional[tuple[float, float]] = None,
            ax: Optional[Union[Axes, NDArrayOfAxes]] = None,
            return_fig: bool = False,
            show: bool = True,
            save: Optional[str] = None,
            **kwargs
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

            cnpl.mad(colorby = "file_name",
                     s = 10,
                     linewidth = 0.4,
                     edgecolor = "black",
                     figsize = (4,4))
        """

        kwargs = self._scatter_defaults(kwargs)

        if data is None:
            mad_frame = self.cnp.mad_frame
        else:
            mad_frame = data

        df = self._prepare_evaluation_frame(dataframe = mad_frame,
                                            file_name = file_name,
                                            channels = channels,
                                            labels = labels)
        df["change"] = (df["original"] - df["normalized"]) < 0
        df["change"] = df["change"].map(
            {False: "decreased", True: "increased"}
        )

        self._check_grid_appropriate(df, grid)

        if grid is not None:
            fig, ax = self._generate_scatter_grid(
                df = df,
                colorby = colorby,
                grid_by = grid,
                grid_n_cols = grid_n_cols,
                figsize = figsize,
                **kwargs
            )
            ax_shape = ax.shape
            ax = ax.flatten()
            for i, _ in enumerate(ax):
                if not ax[i].axison:
                    continue
                # we plot a line to compare the MAD values
                self._draw_cutoff_line(ax[i], cutoff = mad_cutoff)
                ax[i].set_title("MAD comparison")

            ax = ax.reshape(ax_shape)

        else:
            if ax is None:
                if figsize is None:
                    figsize = (2,2)
                fig, ax = plt.subplots(ncols = 1,
                                       nrows = 1,
                                       figsize = figsize)
            else:
                fig = None,
                ax = ax
            assert ax is not None

            plot_kwargs = {
                "data": df,
                "x": "normalized",
                "y": "original",
                "hue": colorby,
                "ax": ax
            }
            assert isinstance(ax, Axes)
            sns.scatterplot(**plot_kwargs,
                            **kwargs)
            self._draw_cutoff_line(ax, cutoff = mad_cutoff)
            ax.set_title("MAD comparison")
            if colorby is not None:
                ax.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left")

        return self._save_or_show(
            ax = ax,
            fig = fig,
            save = save,
            show = show,
            return_fig = return_fig
        )


    def histogram(self,
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
                  figsize: Optional[tuple[float, float]] = None,
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
                           x_channel = "Ho165Di",
                           x_scale = "linear",
                           y_scale = "linear",
                           figsize = (4,4))

        """
        if x_channel is None and grid is None:
            raise ValueError(
                "Either provide a gate or set 'grid' to 'channels'"
            )
        if grid == "file_name":
            raise NotImplementedError("Currently not supported")
            # raise ValueError("A Grid by file_name needs a x_channel")
        if grid == "channels" and file_name is None:
            raise ValueError("A Grid by channels needs a file_name")

        data = self._prepare_data(file_name,
                                  display_reference,
                                  channels,
                                  subsample = subsample)

        kde_kwargs = {}
        hues = data.index.get_level_values("origin").unique().sort_values()
        if grid is not None:
            assert grid == "channels"
            n_cols, n_rows, figsize = self._get_grid_sizes_channels(
                df = data,
                grid_n_cols = grid_n_cols,
                figsize = figsize
            )

            # calculate it to remove empty axes later
            total_plots = n_cols * n_rows

            ax: NDArrayOfAxes
            fig, ax = plt.subplots(
                ncols = n_cols,
                nrows = n_rows,
                figsize = figsize,
                sharex = False,
                sharey = False
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
                    "ax": ax[i]
                }
                ax[i] = sns.kdeplot(**plot_kwargs,
                                    **kde_kwargs,
                                    **kwargs)

                self._handle_axis(ax = ax[i],
                                  x_scale = x_scale,
                                  y_scale = y_scale,
                                  xlim = xlim,
                                  ylim = ylim,
                                  linthresh = linthresh)
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

            fig.legend(
                handles,
                labels,
                bbox_to_anchor = (1.01, 0.5),
                loc = "center left",
                title = "origin"
            )


        else:
            plot_kwargs = {
                "data": data,
                "hue": "origin",
                "hue_order": hues,
                "x": x_channel,
                "ax": ax
            }
            if ax is None:
                if figsize is None:
                    figsize = (2,2)
                fig, ax = plt.subplots(ncols = 1,
                                       nrows = 1,
                                       figsize = figsize)
            else:
                fig = None,
                ax = ax
            assert ax is not None

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

        return self._save_or_show(
            ax = ax,
            fig = fig,
            save = save,
            show = show,
            return_fig = return_fig
        )

    def scatter(self,
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
        subsample
            A number of events to subsample to. Can prevent
            overcrowding of the plot.
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
                         x_channel = "Ho165Di",
                         y_channel = "Yb172Di",
                         x_scale = "linear",
                         y_scale = "linear",
                         figsize = (4,4),
                         s = 10,
                         linewidth = 0.4,
                         edgecolor = "black")


        """

        data = self._prepare_data(file_name,
                                  display_reference,
                                  channels = None,
                                  subsample = subsample)

        if ax is None:
            fig, ax = plt.subplots(ncols = 1,
                                   nrows = 1,
                                   figsize = figsize)
        else:
            fig = None,
            ax = ax
        assert ax is not None
        
        hues = data.index.get_level_values("origin").unique().sort_values()
        plot_kwargs = {
            "data": data,
            "hue": "origin",
            "hue_order": hues,
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

        return self._save_or_show(
            ax = ax,
            fig = fig,
            save = save,
            show = show,
            return_fig = return_fig
        )

    def splineplot(self,
                   file_name: str,
                   channel: str,
                   label_quantiles: Optional[list[float]] = [0.1, 0.25, 0.5, 0.75, 0.9],  # noqa
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
                            channel = "Tb159Di",
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
        ax.set_title(channel)
        self._handle_axis(ax = ax,
                          x_scale = x_scale,
                          y_scale = y_scale,
                          xlim = xlim,
                          ylim = ylim,
                          linthresh = linthresh)

        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        xmin, xmax = ax.get_xlim()
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
            plt.text(x = xmin + 0.01*(xmax-xmin),
                     y = df.loc[df.index == q, "goal"].iloc[0] + ((ylims[1] - ylims[0]) / 200),
                     s = f"Q{int(q*100)}")

        return self._save_or_show(
            ax = ax,
            fig = fig,
            save = save,
            show = show,
            return_fig = return_fig
        )

    def _draw_comp_line(self,
                        ax: Axes) -> None:
        ax.set_ylim(ax.get_xlim())
        comp_line_x = list(ax.get_xlim())
        comp_line_y = comp_line_x
        ax.plot(comp_line_x, comp_line_y, color = "red", linestyle = "--")
        ax.set_xlim(comp_line_x[0], comp_line_x[1])
        ax.set_ylim(comp_line_x[0], comp_line_x[1])
        return

    def _draw_cutoff_line(self,
                          ax: Axes,
                          cutoff: float) -> None:
        ax.set_ylim(ax.get_xlim())
        upper_bound_x = list(ax.get_xlim())
        upper_bound_y = [val + cutoff for val in upper_bound_x]
        lower_bound_x = list(ax.get_ylim())
        lower_bound_y = [val - cutoff for val in lower_bound_x]
        ax.plot(upper_bound_x, upper_bound_y, color = "red", linestyle = "--")
        ax.plot(upper_bound_x, lower_bound_y, color = "red", linestyle = "--")
        ax.set_xlim(upper_bound_x[0], upper_bound_x[1])
        ax.set_ylim(upper_bound_x[0], upper_bound_x[1])

    def _check_grid_appropriate(self,
                                df: pd.DataFrame,
                                grid_by: Optional[str]) -> None:
        if grid_by is not None:
            if df[grid_by].nunique() == 1:
                error_msg = "Only one unique value for the grid variable. "
                error_msg += "A Grid is not possible."
                raise ValueError(error_msg)
        return

    def _get_grid_sizes_channels(self,
                                 df: pd.DataFrame,
                                 grid_n_cols: Optional[int],
                                 figsize: Optional[tuple[float, float]]) -> tuple:

        n_plots = len(df.columns)
        if grid_n_cols is None:
            n_cols = int(np.ceil(np.sqrt(n_plots)))
        else:
            n_cols = grid_n_cols

        n_rows = int(np.ceil(n_plots / n_cols))

        if figsize is None:
            figsize = (3*n_cols, 3*n_rows)

        return n_cols, n_rows, figsize

    def _get_grid_sizes(self,
                        df: pd.DataFrame,
                        grid_by: str,
                        grid_n_cols: Optional[int],
                        figsize: Optional[tuple[float, float]]) -> tuple:

        n_plots = df[grid_by].nunique()
        if grid_n_cols is None:
            n_cols = int(np.ceil(np.sqrt(n_plots)))
        else:
            n_cols = grid_n_cols

        n_rows = int(np.ceil(n_plots / n_cols))

        if figsize is None:
            figsize = (3*n_cols, 3*n_rows)

        return n_cols, n_rows, figsize

    def _generate_scatter_grid(self,
                               df: pd.DataFrame,
                               grid_by: str,
                               grid_n_cols: Optional[int],
                               figsize: tuple[float, float],
                               colorby: Optional[str],
                               **scatter_kwargs: Optional[dict]
                               ) -> tuple[Figure, NDArrayOfAxes]:

        n_cols, n_rows, figsize = self._get_grid_sizes(
            df = df,
            grid_by = grid_by,
            grid_n_cols = grid_n_cols,
            figsize = figsize
        )

        # calculate it to remove empty axes later
        total_plots = n_cols * n_rows
        
        hue = None if colorby == grid_by else colorby
        plot_params = {
            "x": "normalized",
            "y": "original",
            "hue": hue
        }

        fig, ax = plt.subplots(
            ncols = n_cols,
            nrows = n_rows,
            figsize = figsize,
            sharex = True,
            sharey = True
        )
        ax = ax.flatten()
        i = 0

        for i, grid_param in enumerate(df[grid_by].unique()):
            sns.scatterplot(
                data = df[df[grid_by] == grid_param],
                **plot_params,
                **scatter_kwargs,
                ax = ax[i]
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
            fig.legend(
                handles,
                labels,
                bbox_to_anchor = (1.01, 0.5),
                loc = "center left",
                title = colorby
            )

        return fig, ax

    def _scatter_defaults(self,
                          kwargs: dict) -> dict:
        kwargs["s"] = kwargs.get("s", 2)
        kwargs["edgecolor"] = kwargs.get("edgecolor", "black")
        kwargs["linewidth"] = kwargs.get("linewidth", 0.1)
        return kwargs

    def _prepare_evaluation_frame(self,
                                  dataframe: pd.DataFrame,
                                  file_name: Optional[Union[list[str], str]] = None,
                                  channels: Optional[Union[list[str], str]] = None,
                                  labels: Optional[Union[list[str], str]] = None) -> pd.DataFrame:
        index_names = dataframe.index.names
        dataframe = dataframe.reset_index()
        melted = dataframe.melt(id_vars = index_names,
                                var_name = "channel",
                                value_name = "value")
        df = melted.pivot_table(index = [
                                    idx_name
                                    for idx_name in index_names
                                    if idx_name != "origin"
                                ] + ["channel"],
                                columns = "origin",
                                values = "value").reset_index()
        if file_name is not None:
            if not isinstance(file_name, list):
                file_name = [file_name]
            df = df.loc[df["file_name"].isin(file_name),:]

        if channels is not None:
            if not isinstance(channels, list):
                channels = [channels]
            df = df.loc[df["channel"].isin(channels),:]

        if labels is not None:
            if not isinstance(labels, list):
                labels = [labels]
            df = df.loc[df["label"].isin(labels),:]

        return df


    def _select_index_levels(self,
                             df: pd.DataFrame):
        index_levels_to_keep = ["origin", "reference", "batch", "file_name"]
        for name in df.index.names:
            if name not in index_levels_to_keep:
                df = df.droplevel(name)
        return df

    def _prepare_data(self,
                      file_name: str,
                      display_reference: bool,
                      channels: Optional[Union[list[str], str]],
                      subsample: Optional[int]
                      ) -> pd.DataFrame:

        original_df = self.cnp._datahandler \
            .get_dataframe(file_name)

        normalized_df = self.cnp.\
            _normalize_file(
                df = original_df.copy(),
                batch = self.cnp._datahandler.get_batch(file_name)
            )

        if display_reference is True:
            ref_df = self.cnp._datahandler \
                .get_corresponding_ref_dataframe(file_name)
            ref_df["origin"] = "reference"
            ref_df = ref_df.set_index("origin", append = True, drop = True)
            ref_df = self._select_index_levels(ref_df)
        else:
            ref_df = None

        original_df["origin"] = "original"
        normalized_df["origin"] = "transformed"

        original_df = original_df.set_index("origin", append = True, drop = True)
        normalized_df = normalized_df.set_index("origin", append = True, drop = True)

        original_df = self._select_index_levels(original_df)
        normalized_df = self._select_index_levels(normalized_df)

        # we clean up the indices in order to not mess up the

        if ref_df is not None:
            data = pd.concat([normalized_df,
                              original_df,
                              ref_df], axis = 0)
        else:
            data = pd.concat([normalized_df,
                              original_df], axis = 0)

        if channels is not None:
            data = data[channels]

        if subsample:
            data = data.sample(n = subsample)
        else:
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
