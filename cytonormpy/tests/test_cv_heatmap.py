import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import cytonormpy as cnp


def test_cv_heatmap_precomputed_fig():
    cn = cnp.CytoNorm()
    cn.cvs_by_k = {
        2: [0.1, 1.6],
        3: [1.0, 0.0, 2.6],
    }
    ks = [2, 3]

    fig = cnp.pl.cv_heatmap(
        cnp=cn,
        n_metaclusters=ks,
        max_cv=2.5,
        show_cv=1.5,
        return_fig=True,
        show=False,
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]

    images = ax.get_images()
    assert len(images) == 1
    arr = images[0].get_array()

    assert arr.shape == (2, 3)
    assert pytest.approx(arr[1, 2]) == 2.5
    assert pytest.approx(arr[0, 1]) == 1.6

    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    assert ylabels == ["2", "3"]

    texts = {t.get_text() for t in ax.texts}
    assert "1.60" in texts
    assert "2.60" in texts
    assert "1.00" not in texts


def test_cv_heatmap_return_axes_and_no_texts():
    cn = cnp.CytoNorm()
    cn.cvs_by_k = {1: [0.2], 2: [0.0, 0.4]}
    ks = [1, 2]

    ax = cnp.pl.cv_heatmap(
        cnp=cn,
        n_metaclusters=ks,
        max_cv=1.0,
        show_cv=0.5,
        return_fig=False,
        show=False,
    )
    assert isinstance(ax, Axes)

    arr = ax.get_images()[0].get_array()
    assert arr.shape == (2, 2)

    assert len(ax.texts) == 0


def test_cv_heatmap_auto_compute(monkeypatch):
    cn = cnp.CytoNorm()

    def fake_calc(self, ks):
        self.cvs_by_k = {k: [float(i) for i in range(k)] for k in ks}

    monkeypatch.setattr(cnp.CytoNorm, "calculate_cluster_cvs", fake_calc)

    ks = [3]
    fig = cnp.pl.cv_heatmap(cnp=cn, n_metaclusters=ks, return_fig=True, show=False)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    arr = ax.get_images()[0].get_array()
    assert arr.shape == (1, 3)
    assert np.allclose(arr[0, :], [0.0, 1.0, 2.0])
