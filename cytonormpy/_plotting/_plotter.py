import warnings

from ._scatter import scatter as scatter_func
from ._histogram import histogram as histogram_func
from ._evaluations import emd as emd_func, mad as mad_func
from ._splineplot import splineplot as splineplot_func


class Plotter:
    """
    Deprecated wrapper for plotting functions.

    Raises a DeprecationWarning upon creation; all methods forward
    arguments to the module-level plotting functions.
    """

    def __init__(self, cytonorm):
        warnings.warn(
            "Plotter is deprecated; use the standalone plotting functions "
            "(e.g. cnp.pl.scatter, cnp.pl.histogram, cnp.pl.emd, cnp.pl.mad, cnp.pl.splineplot) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cnp = cytonorm

    def scatter(self, *args, **kwargs):
        return scatter_func(self.cnp, *args, **kwargs)

    def histogram(self, *args, **kwargs):
        return histogram_func(self.cnp, *args, **kwargs)

    def emd(self, *args, **kwargs):
        return emd_func(self.cnp, *args, **kwargs)

    def mad(self, *args, **kwargs):
        return mad_func(self.cnp, *args, **kwargs)

    def splineplot(self, *args, **kwargs):
        return splineplot_func(self.cnp, *args, **kwargs)
