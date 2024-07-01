import numpy as np
from typing import Union, Callable, Literal, Optional

from scipy.interpolate import CubicHermiteSpline, PPoly

from .._utils._utils import regularize_values, _select_interpolants_numba


class IdentitySpline:
    """\
    Class to serve as a replacement if no spline
    function can be calculated.

    Only implements __call__ method that returns
    the data unmodified.

    """

    def __init__(self):
        pass

    def __call__(self,
                 data: np.ndarray) -> np.ndarray:
        return data


class Spline:
    """\
    Class to perform the spline calculations.

    Parameters
    ----------
    batch
        The batch of which the spline functions are
        calculated.
    cluster
        The cluster of which the spline function is
        calculated.
    channel
        The channel that is calculated
    spline_calc_function
        The spline function that is used for the spline
        calculation. Defaults to CubicHermiteSpline as
        in the original CytoNorm implementation in R.
    extrapolate
        How to extrapolate data points outside of the
        spline range. Defaults to `linear` as implemented
        in the R stats::splinefun() function. If `False`,
        values are not extrapolated and result in NaN
        values. If `spline`, use the spline function
        for extrapolation.
    limits
        A list or array of fixed intensity values. These values will be
        appended to the calculated quantiles and included to calculate
        the spline functions. By default, the spline functions are extrapolated
        linearly outside the observed data range, using the slope at the last
        data point as the slope for the extrapolation. Use `limits` to further
        control the behaviour outside the data range.

    """
    def __init__(self,
                 batch: Union[float, str],
                 cluster: Union[float, str],
                 channel: str,
                 spline_calc_function: Callable = CubicHermiteSpline,
                 extrapolate: Union[Literal["linear", "spline"], bool] = "linear", # noqa
                 limits: Optional[Union[list[float], np.ndarray]] = None
                 ) -> None:
        self.batch = batch
        self.channel = channel
        self.cluster = cluster
        self.spline_calc_function = spline_calc_function
        self._extrapolate = extrapolate
        self._limits = limits

        if self._limits is not None:
            self._limits = np.array(self._limits)

    def _select_interpolants(self,
                             x: np.ndarray,
                             y: np.ndarray) -> np.ndarray:
        return _select_interpolants_numba(x, y)

    def _append_limits(self,
                       arr: np.ndarray) -> np.ndarray:
        if self._limits is None:
            return arr
        return np.hstack([arr, self._limits])

    def fit(self,
            current_distribution: Optional[np.ndarray],
            goal_distribution: Optional[np.ndarray],
            ) -> None:
        """\
        Interpolates a function between the current expression values
        and the goal expression values. First, limits are appended
        and the arrays are regularized  in order to prevent duplicate
        entries. Next, interpolants are selected using the Fritsch-Carlson
        algorithm. The Cubic Hermite Spline is fitted and extrapolated
        linearly outside of the data range.

        Parameters
        ----------
        current_distribution
            A numpy array containing the expression values
            at the quantiles
        goal_distribution
            A numpy array containing the goal expression
            values at the quantiles

        Returns
        -------
        None


        """

        if self.spline_calc_function.__qualname__ == "IdentitySpline":
            self.fit_func = self.spline_calc_function()
        else:
            assert not self.is_fit()
            assert current_distribution is not None
            assert goal_distribution is not None

            current_distribution = self._append_limits(current_distribution)
            goal_distribution = self._append_limits(goal_distribution)
            
            current_distribution, goal_distribution = regularize_values(
                current_distribution,
                goal_distribution
            )

            m = self._select_interpolants(
                current_distribution,
                goal_distribution
            )
            self.fit_func: PPoly = self.spline_calc_function(
                current_distribution,
                goal_distribution,
                dydx = m,
                extrapolate = True if self._extrapolate is not False else False
            )
            if self._extrapolate == "linear":
                self._extrapolate_linear()

    def _extrapolate_linear(self) -> None:
        """\
        Extrapolation is achieved linearly using the slope
        of the CubicHermiteSpline at the nearest data point.
        """
        # determine the slope at the left edge
        leftx = self.fit_func.x[0]
        lefty = self.fit_func(leftx)
        leftslope = self.fit_func(leftx, nu=1)
        leftxnext = np.nextafter(leftx, leftx - 1)
        leftynext = lefty + leftslope * (leftxnext - leftx)
        leftcoeffs = np.array([0, 0, leftslope, leftynext])
        self.fit_func.extend(leftcoeffs[..., None], np.r_[leftxnext])

        # repeat with additional knots to the right
        rightx = self.fit_func.x[-1]
        righty = self.fit_func(rightx)
        rightslope = self.fit_func(rightx, nu=1)
        rightxnext = np.nextafter(rightx, rightx + 1)
        rightynext = righty + rightslope * (rightxnext - rightx)
        rightcoeffs = np.array([0, 0, rightslope, rightynext])
        self.fit_func.extend(rightcoeffs[..., None], np.r_[rightxnext])

    def transform(self,
                  distribution: np.ndarray) -> np.ndarray:
        """\
        Calculates new expression values based on the spline function.

        Parameters
        ----------
        distribution
            A numpy array containing the current expression values

        Returns
        -------
        A :class:`np.ndarray` with the corrected expression values.

        """
        return self.fit_func(distribution)

    def is_fit(self):
        return hasattr(self, "fit_func")


class Splines:
    """\
    Class to hold splines in a dictionary of shape

    {batch: {cluster: {channel: splinefunc, ...}}}

    """

    def __init__(self,
                 batches: list[Union[float, str]],
                 clusters: list[Union[float, str]],
                 channels: list[Union[float, str]]) -> None:
        self._init_dictionary(batches, clusters, channels)

    def _init_dictionary(self,
                         batches: list[Union[float, str]],
                         clusters: list[Union[float, str]],
                         channels: list[Union[float, str]]) -> None:
        """\
        Instantiates the dictionary.

        Parameters
        ----------
        batches
            All the batches as a list.
        clusters
            All the clusters as a list.
        channels
            All the channels as a list.

        Returns
        -------
        None

        """
        self._splines: dict = {
            batch:
                {cluster:
                    {channel: None
                     for channel in channels}
                 for cluster in clusters}
            for batch in batches
        }

    def add_spline(self,
                   spline: Spline) -> None:
        """\
        Adds the spline function according to from the dict
        according to batch, cluster and channel.

        Parameters
        ----------
        spline
            Object of type Spline, containing the spline function.

        Returns
        -------
        None

        """
        assert spline.is_fit(), "Please fit first."
        batch = spline.batch
        cluster = spline.cluster
        channel = spline.channel
        self._splines[batch][cluster][channel] = spline

    def remove_spline(self,
                      batch: Union[float, str],
                      cluster: Union[float, str],
                      channel: Union[float, str]) -> None:
        """\
        Deletes the spline function according to from the dict
        according to batch, cluster and channel.

        Parameters
        ----------
        batch
            The batch where the input data are originating from.
        cluster
            The cluster that the data are originating from.
        channel
            The channel that the data are originating from.

        Returns
        -------
        None

        """
        del self._splines[batch][cluster][channel]

    def get_spline(self,
                   batch: Union[float, str],
                   cluster: Union[float, str],
                   channel: str) -> Spline:
        """\
        Returns the correct spline function according to
        batch, cluster and channel.

        Parameters
        ----------
        batch
            The batch where the input data are originating from.
        cluster
            The cluster that the data are originating from.
        channel
            The channel that the data are originating from.

        Returns
        -------
        A Spline object.

        """
        return self._splines[batch][cluster][channel]

    def transform(self,
                  data: np.ndarray,
                  batch: Union[float, str],
                  cluster: Union[float, str],
                  channel: str) -> np.ndarray:
        """\
        Extracts the correct spline function according to
        batch, cluster and channel and returns the corrected
        data according to this spline function.

        Parameters
        ----------
        data
            The data to transform.
        batch
            The batch where the input data are originating from.
        cluster
            The cluster that the data are originating from.
        channel
            The channel that the data are originating from.

        Returns
        -------
        A numpy array with the corrected expression values.

        """
        req_spline: Spline = self.get_spline(batch = batch,
                                             cluster = cluster,
                                             channel = channel)
        return req_spline.transform(data)
