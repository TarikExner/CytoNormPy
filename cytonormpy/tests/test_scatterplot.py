import pytest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from types import SimpleNamespace
import cytonormpy as cnp


class DummyDataHandlerScatter:
    """Minimal DataHandler stub for scatter tests."""

    def __init__(self):
        self.metadata = SimpleNamespace(get_batch=lambda file_name: "batch1")

    def get_dataframe(self, file_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "X": [0, 1, 2, 3],
                "Y": [3, 2, 1, 0],
            }
        )

    def get_corresponding_ref_dataframe(self, file_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "X": [10],
                "Y": [10],
            }
        )


@pytest.fixture(autouse=True)
def no_gui(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def test_scatter_basic_axes_and_scatter_count(monkeypatch):
    cn = cnp.CytoNorm()
    cn._datahandler = DummyDataHandlerScatter()
    cn._normalize_file = lambda df, batch: df

    ax = cnp.pl.scatter(
        cnp=cn,
        file_name="any.fcs",
        x_channel="X",
        y_channel="Y",
        x_scale="linear",
        y_scale="linear",
        display_reference=False,  # skip reference for this test
        return_fig=False,
        show=False,
    )
    assert isinstance(ax, Axes)

    pcs = [c for c in ax.get_children() if isinstance(c, PathCollection)]
    assert len(pcs) >= 1

    # total number of points should be 4+4 = 8, because we do not show the ref.
    total_points = sum(pc.get_offsets().shape[0] for pc in pcs)
    assert total_points == 8

    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x0 <= 0 and x1 >= 3
    assert y0 <= 0 and y1 >= 3

    leg = ax.get_legend()
    assert leg is not None
    texts = [t.get_text() for t in leg.get_texts()]
    assert set(texts) == {"original", "transformed"}


def test_scatter_with_reference_and_return_fig(monkeypatch):
    cn = cnp.CytoNorm()
    cn._datahandler = DummyDataHandlerScatter()
    cn._normalize_file = lambda df, batch: df

    fig = cnp.pl.scatter(
        cnp=cn,
        file_name="any.fcs",
        x_channel="X",
        y_channel="Y",
        x_scale="log",
        y_scale="log",
        display_reference=True,
        return_fig=True,
        show=False,
    )
    assert isinstance(fig, Figure)

    axes = fig.get_axes()
    assert len(axes) == 1
    ax = axes[0]

    # Collect all PathCollections that represent scatter layers
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(pcs) >= 1  # at least one scatter layer

    # Total number of plotted points should be 9 (4 orig + 4 trans + 1 ref)
    total = sum(pc.get_offsets().shape[0] for pc in pcs)
    assert total == 9

    # Check log scales
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"

    # Legend should now include "reference" as well
    leg = ax.get_legend()
    labels = [t.get_text() for t in leg.get_texts()]
    assert set(labels) == {"original", "transformed", "reference"}


def test_scatter_custom_legend_labels(monkeypatch):
    cn = cnp.CytoNorm()
    cn._datahandler = DummyDataHandlerScatter()
    cn._normalize_file = lambda df, batch: df

    custom = ["A", "B"]
    ax = cnp.pl.scatter(
        cnp=cn,
        file_name="any.fcs",
        x_channel="X",
        y_channel="Y",
        legend_labels=custom,
        display_reference=False,
        return_fig=False,
        show=False,
    )

    leg = ax.get_legend()
    labels = [t.get_text() for t in leg.get_texts()]
    assert labels == custom
