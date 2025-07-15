import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import cytonormpy._plotting._histogram as hist_module
from cytonormpy._plotting._histogram import histogram as histfunc

import cytonormpy as cnp


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    def fake_prepare(cnp_obj, file_name, display_reference, channels, subsample):
        origins = ["original"] * 50 + ["transformed"] * 50
        return pd.DataFrame(
            {
                "A": np.concatenate([np.zeros(50), np.ones(50)]),
                "B": np.concatenate([np.ones(50), np.zeros(50)]),
            },
            index=pd.Index(origins, name="origin"),
        )

    monkeypatch.setattr(hist_module, "_prepare_data", fake_prepare)

    def fake_modify_axes(ax, x_scale, y_scale, xlim, ylim, linthresh):
        # treat 'biex' as linear for test purposes
        ax.set_xscale("linear" if x_scale == "biex" else x_scale)
        ax.set_yscale("linear" if y_scale == "biex" else y_scale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    monkeypatch.setattr(hist_module, "modify_axes", fake_modify_axes)

    monkeypatch.setattr(
        hist_module,
        "save_or_show",
        lambda *, ax, fig, save, show, return_fig: (fig if return_fig else ax),
    )


def test_histogram_requires_args():
    cn = cnp.CytoNorm()
    with pytest.raises(ValueError):
        histfunc(cnp=cn, file_name="f", x_channel=None, grid=None, show=False)
    with pytest.raises(NotImplementedError):
        histfunc(cnp=cn, file_name="f", grid="file_name", x_channel="A", show=False)
    with pytest.raises(ValueError):
        histfunc(cnp=cn, file_name=None, grid="channels", x_channel=None, show=False)


def test_histogram_basic_density():
    cn = cnp.CytoNorm()
    ax = histfunc(cnp=cn, file_name="f", x_channel="A", grid=None, return_fig=False, show=False)
    assert isinstance(ax, Axes)

    leg = ax.get_legend()
    assert leg is not None
    texts = {t.get_text() for t in leg.get_texts()}
    assert texts == {"original", "transformed"}

    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"

    x0, x1 = ax.get_xlim()
    assert x0 <= 0 and x1 >= 1


def test_histogram_return_fig_log_scales():
    cn = cnp.CytoNorm()
    fig = histfunc(
        cnp=cn,
        file_name="f",
        x_channel="B",
        grid=None,
        x_scale="log",
        y_scale="log",
        return_fig=True,
        show=False,
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"

    leg = ax.get_legend()
    texts = {t.get_text() for t in leg.get_texts()}
    assert texts == {"original", "transformed"}


def test_histogram_channels_grid_layout():
    cn = cnp.CytoNorm()
    fig = histfunc(cnp=cn, file_name="f", grid="channels", return_fig=True, show=False)
    assert isinstance(fig, Figure)
    axes = fig.axes

    assert len(axes) == 2

    titles = {ax.get_title() for ax in axes}
    assert titles == {"A", "B"}

    legends = fig.legends
    assert len(legends) == 1
    legend_texts = {t.get_text() for t in legends[0].get_texts()}
    assert legend_texts == {"original", "transformed"}


def test_histogram_custom_grid_n_cols():
    cn = cnp.CytoNorm()
    fig = histfunc(
        cnp=cn, file_name="f", grid="channels", grid_n_cols=1, return_fig=True, show=False
    )
    axes = fig.axes
    assert len(axes) == 2
    assert axes[0].get_title() == "A"
    assert axes[1].get_title() == "B"
