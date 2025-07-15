import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import cytonormpy._plotting._utils as utils


def test_set_scatter_defaults_empty():
    kwargs = {}
    out = utils.set_scatter_defaults(kwargs.copy())
    assert out["s"] == 2
    assert out["edgecolor"] == "black"
    assert out["linewidth"] == 0.1


def test_set_scatter_defaults_override():
    kwargs = {"s": 10, "edgecolor": "red"}
    out = utils.set_scatter_defaults(kwargs.copy())
    assert out["s"] == 10
    assert out["edgecolor"] == "red"
    assert out["linewidth"] == 0.1


def test_modify_legend_default_and_custom():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="first")
    ax.plot([0, 1], [1, 0], label="second")
    ax.legend()
    utils.modify_legend(ax, legend_labels=None)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["first", "second"]
    custom = ["A", "B"]
    utils.modify_legend(ax, legend_labels=custom)
    texts2 = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts2 == custom
    plt.close(fig)


@pytest.mark.parametrize(
    "x_scale,y_scale,expected_x,expected_y",
    [
        ("linear", "linear", "linear", "linear"),
        ("log", "log", "log", "log"),
        ("biex", "linear", "symlog", "linear"),
        ("linear", "biex", "linear", "symlog"),
        ("biex", "biex", "symlog", "symlog"),
    ],
)
def test_modify_axes_scales_and_limits(x_scale, y_scale, expected_x, expected_y):
    fig, ax = plt.subplots()
    utils.modify_axes(
        ax=ax,
        x_scale=x_scale,
        y_scale=y_scale,
        linthresh=0.5,
        xlim=(1, 3),
        ylim=(2, 4),
    )
    assert ax.get_xscale() == expected_x
    assert ax.get_yscale() == expected_y
    assert ax.get_xlim() == (1, 3)
    assert ax.get_ylim() == (2, 4)
    plt.close(fig)


def test_save_or_show_behaviors(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    saved = {}
    monkeypatch.setattr(plt, "savefig", lambda fname, **kw: saved.setdefault("file", fname))
    monkeypatch.setattr(plt, "show", lambda **kw: saved.setdefault("shown", True))

    out1 = utils.save_or_show(ax=ax, fig=fig, save=None, show=False, return_fig=False)
    assert isinstance(out1, Axes)
    assert "shown" not in saved

    out2 = utils.save_or_show(ax=ax, fig=fig, save=None, show=False, return_fig=True)
    assert isinstance(out2, Figure)

    fp = str(tmp_path / "out.png")
    _ = utils.save_or_show(ax=ax, fig=fig, save=fp, show=False, return_fig=False)
    assert saved["file"] == fp

    out4 = utils.save_or_show(ax=ax, fig=fig, save=None, show=True, return_fig=False)
    assert out4 is None
    assert saved.get("shown", False) is True

    plt.close(fig)
