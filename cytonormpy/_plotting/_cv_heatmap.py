import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Union

from .._cytonorm import CytoNorm
from ._utils import save_or_show


def cv_heatmap(
    cnp: CytoNorm,
    n_metaclusters: list[int],
    max_cv: float = 2.5,
    show_cv: float = 1.5,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 4),
    ax: Optional[Axes] = None,
    return_fig: bool = False,
    show: bool = True,
    save: Optional[str] = None,
) -> Optional[Union[Figure, Axes]]:
    """
    Plot a heatmap of cluster CVs for a set of meta‑cluster counts.

    Parameters
    ----------
    cnp
        A CytoNorm instance that has run calculate_cluster_cvs.
    n_metaclusters
        List of meta‑cluster counts whose CVs you wish to plot.
    max_cv
        Clip color scale at this CV value.
    show_cv
        Only CVs >= show_cv get a numeric label.
    cmap
        Name of the matplotlib colormap to use.
    figsize
        Figure size, used only if ax is None.
    ax
        Optional Axes to draw into. If None, a new Figure+Axes is created.
    return_fig
        If True, return the Figure; otherwise, return the Axes.
    show
        If True, call plt.show() at the end.
    save
        File path to save the figure. If None, no file is written.

    Returns
    -------
    Figure or Axes or None
        If `return_fig`, returns the Figure; else returns the Axes.
        If both are False, returns None.

    Examples
    --------
    .. plot::
        :context: close-figs

        import cytonormpy as cnp

        cn = cnp.example_cytonorm(use_clustering = True)
        cn.calculate_cluster_cvs(n_metaclusters = list(range(3,15)))
        cnp.pl.cv_heatmap(cn,
                          n_metaclusters = list(range(3,15)),
                          max_cv = 2,
                          figsize = (4,3)
                          )
    """
    if not hasattr(cnp, "cvs_by_k"):
        cnp.calculate_cluster_cvs(n_metaclusters)

    cvs_dict = cnp.cvs_by_k
    ks = n_metaclusters
    max_k = max(ks)

    mat = np.full((len(ks), max_k), np.nan, dtype=float)
    for i, k in enumerate(ks):
        row = np.array(cvs_dict[k], dtype=float)
        mat[i, : len(row)] = row

    text = np.full(mat.shape, "", dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v) and v >= show_cv:
                text[i, j] = f"{v:.2f}"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        ax = ax

    assert ax is not None
    assert fig is not None

    im = ax.imshow(
        np.clip(mat, 0, max_cv),
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=max_cv,
        cmap=cmap,
    )
    for (i, j), label in np.ndenumerate(text):
        if label:
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="white")

    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("Cluster index")
    ax.set_ylabel("Meta‑cluster count")

    fig.colorbar(im, ax=ax, label="CV")

    return save_or_show(ax=ax, fig=fig, save=save, show=show, return_fig=return_fig)
