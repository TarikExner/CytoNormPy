import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from types import SimpleNamespace

import cytonormpy._plotting._splineplot as spl_module
from cytonormpy._plotting._splineplot import splineplot

import cytonormpy as cnp


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    # Prevent plt.show() from blocking
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    # Stub modify_axes so it applies scales & limits
    def fake_modify_axes(ax, x_scale, y_scale, xlim, ylim, linthresh):
        ax.set_xscale("linear" if x_scale == "biex" else x_scale)
        ax.set_yscale("linear" if y_scale == "biex" else y_scale)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    monkeypatch.setattr(spl_module, "modify_axes", fake_modify_axes)

    # Stub save_or_show
    monkeypatch.setattr(
        spl_module,
        "save_or_show",
        lambda *, ax, fig, save, show, return_fig: (fig if return_fig else ax),
    )


def make_dummy_cnp():
    """Return a CytoNorm with minimal attrs for splineplot."""
    cn = cnp.CytoNorm()

    class DummyEQ:
        def __init__(self):
            self.quantiles = np.array([0.1, 0.5, 0.9])
            self._cluster_axis = 0

        def get_quantiles(self, channel_idx, batch_idx, cluster_idx, quantile_idx, flattened):
            # shape (1, n_quantiles)
            return np.array([self.quantiles])

    class DummyGD:
        def __init__(self, quantiles):
            # give it the same .quantiles so code won't break
            self.quantiles = quantiles

        def get_quantiles(self, channel_idx, batch_idx, cluster_idx, quantile_idx, flattened):
            # return twice the expr quantiles
            return np.array([quantiles * 2.0])

    # instantiate expr & goal
    eq = DummyEQ()
    quantiles = eq.quantiles
    gd = DummyGD(quantiles)

    cn._expr_quantiles = eq
    cn._goal_distrib = gd
    cn.batches = ["batchA"]
    cn.channels = ["ch1"]
    cn._datahandler = SimpleNamespace(metadata=SimpleNamespace(get_batch=lambda fn: "batchA"))
    return cn


def test_splineplot_basic_line_and_text():
    cn = make_dummy_cnp()
    qs = [0.1, 0.9]
    ax = splineplot(
        cnp=cn,
        file_name="any.fcs",
        channel="ch1",
        label_quantiles=qs,
        x_scale="log",
        y_scale="log",
        return_fig=False,
        show=False,
    )
    assert isinstance(ax, Axes)
    assert ax.get_title() == "ch1"
    lines = ax.get_lines()
    assert len(lines) == 1
    # one vertical+one horizontal per quantile
    # but each quantile adds 2 Line2D, we only care about text labels count
    assert len(ax.texts) == len(qs)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_splineplot_return_fig():
    cn = make_dummy_cnp()
    fig = splineplot(
        cnp=cn,
        file_name="any.fcs",
        channel="ch1",
        label_quantiles=[0.5],
        return_fig=True,
        show=False,
    )
    assert isinstance(fig, Figure)
    axes = fig.get_axes()
    assert len(axes) == 1
    assert axes[0].get_title() == "ch1"


def test_splineplot_custom_limits_and_no_labels():
    cn = make_dummy_cnp()
    ax = splineplot(
        cnp=cn,
        file_name="any.fcs",
        channel="ch1",
        label_quantiles=None,
        xlim=(2, 4),
        ylim=(5, 10),
        return_fig=False,
        show=False,
    )
    assert isinstance(ax, Axes)
    # no text labels
    assert len(ax.texts) == 0
    # limits applied
    assert ax.get_xlim() == (2, 4)
    assert ax.get_ylim() == (5, 10)
