import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection

import cytonormpy._plotting._evaluations as eval_mod
from cytonormpy._plotting._evaluations import emd, mad
import cytonormpy._plotting._utils as utils_mod

import cytonormpy as cnp


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    # silence plt.show()
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    # Stub out the common helpers in utils
    monkeypatch.setattr(utils_mod, "set_scatter_defaults", lambda kwargs: kwargs)
    monkeypatch.setattr(utils_mod, "modify_axes", lambda *a, **k: None)
    monkeypatch.setattr(utils_mod, "modify_legend", lambda *a, **k: None)

    # Now stub only the private internals in evaluations
    def real_check(df, grid_by):
        if grid_by is not None and df[grid_by].nunique() == 1:
            raise ValueError("Only one unique value for the grid variable. A Grid is not possible.")

    monkeypatch.setattr(eval_mod, "_check_grid_appropriate", real_check)

    monkeypatch.setattr(
        eval_mod, "_prepare_evaluation_frame", lambda dataframe, **kw: dataframe.copy()
    )
    monkeypatch.setattr(eval_mod, "_draw_comp_line", lambda ax: None)
    monkeypatch.setattr(eval_mod, "_draw_cutoff_line", lambda ax, cutoff=None: None)

    def fake_gen(df, grid_by, grid_n_cols, figsize, colorby, **kw):
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        return fig, np.array(axes)

    monkeypatch.setattr(eval_mod, "_generate_scatter_grid", fake_gen)

    monkeypatch.setattr(
        eval_mod,
        "save_or_show",
        lambda *, ax, fig, save, show, return_fig: (fig if return_fig else ax),
    )


def make_emd_df():
    return pd.DataFrame(
        {
            "original": [1.0, 2.0, 3.0, 4.0],
            "normalized": [1.5, 1.5, 2.5, 3.5],
            "label": ["A", "A", "B", "B"],
        }
    )


def make_mad_df():
    return pd.DataFrame(
        {
            "original": [1.0, 0.5, 2.0, 2.5],
            "normalized": [0.5, 1.0, 2.5, 2.0],
            "file_name": ["f1", "f1", "f2", "f2"],
        }
    )


def test_emd_basic_scatter_axes_and_legend():
    df = make_emd_df()
    cn = cnp.CytoNorm()
    ax = emd(cnp=cn, colorby="label", data=df, grid=None, return_fig=False, show=False)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "EMD comparison"
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert pcs, "No scatter collections found"
    texts = {t.get_text() for t in ax.get_legend().get_texts()}
    assert texts == {"A", "B"}


def test_emd_grid_layout_and_legend():
    df = make_emd_df()
    cn = cnp.CytoNorm()
    fig = emd(
        cnp=cn, colorby="label", data=df, grid="label", grid_n_cols=2, return_fig=True, show=False
    )
    assert isinstance(fig, Figure)
    axes = fig.axes
    assert len(axes) == 2
    titles = {ax.get_title() for ax in axes}
    assert titles == {"EMD comparison"}
    legends = fig.legends
    assert len(legends) == 0


def test_emd_grid_error_single_value():
    df = make_emd_df()
    df["label"] = ["A"] * 4
    cn = cnp.CytoNorm()
    with pytest.raises(ValueError):
        emd(cnp=cn, colorby="label", data=df, grid="label", show=False)


def test_mad_basic_scatter_and_legend():
    df = make_mad_df()
    cn = cnp.CytoNorm()
    ax = mad(cnp=cn, colorby="file_name", data=df, grid=None, return_fig=False, show=False)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "MAD comparison"
    texts = {t.get_text() for t in ax.get_legend().get_texts()}
    assert texts == {"f1", "f2"}


def test_mad_grid_layout_and_no_legend():
    df = make_mad_df()
    cn = cnp.CytoNorm()
    fig = mad(
        cnp=cn,
        colorby="file_name",
        data=df,
        grid="file_name",
        grid_n_cols=2,
        return_fig=True,
        show=False,
    )
    assert isinstance(fig, Figure)
    axes = fig.axes
    assert len(axes) == 2
    titles = {ax.get_title() for ax in axes}
    assert titles == {"MAD comparison"}
    assert len(fig.legends) == 0


def test_mad_grid_error_single_value():
    df = make_mad_df()
    df["file_name"] = ["f1"] * 4
    cn = cnp.CytoNorm()
    with pytest.raises(ValueError):
        mad(cnp=cn, colorby="file_name", data=df, grid="file_name", show=False)
