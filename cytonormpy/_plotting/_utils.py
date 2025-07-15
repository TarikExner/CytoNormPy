import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Union

DEFAULT_MARKERS = ["o", "^", "s", "P", "D", "X", "v", "<", ">", "*"]
DASH_STYLES = ["solid", "dashed", "dashdot", "dotted"]


def apply_vary_textures(plot_kwargs: dict, df: pd.DataFrame, hue: Optional[str]) -> None:
    """
    Mutates plot_kwargs in-place to add seaborn-style marker variation
    based on the categories in df[hue].
    """
    if not hue:
        return
    levels = list(df[hue].unique())
    plot_kwargs["style"] = hue
    plot_kwargs["style_order"] = levels
    plot_kwargs["markers"] = {
        lvl: DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)] for i, lvl in enumerate(levels)
    }


def set_scatter_defaults(kwargs: dict) -> dict:
    kwargs["s"] = kwargs.get("s", 2)
    kwargs["edgecolor"] = kwargs.get("edgecolor", "black")
    kwargs["linewidth"] = kwargs.get("linewidth", 0.1)
    return kwargs


def modify_legend(ax: Axes, legend_labels: Optional[list[str]]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if legend_labels:
        labels = legend_labels
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return


def modify_axes(
    ax: Axes,
    x_scale: str,
    y_scale: str,
    linthresh: Optional[float],
    xlim: Optional[tuple[float, float]],
    ylim: Optional[tuple[float, float]],
) -> None:
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

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return


def save_or_show(
    ax: Axes, fig: Optional[Figure], save: Optional[str], show: bool, return_fig: bool
) -> Optional[Union[Figure, Axes]]:
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if return_fig:
        return fig

    return ax if not show else None
