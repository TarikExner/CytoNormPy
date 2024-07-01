import numpy as np
from typing import Union, Optional, Callable

from ._utils import numba_quantiles

class BaseQuantileHandler:

    def __init__(self,
                 channel_axis: int,
                 quantile_axis: int,
                 cluster_axis: int,
                 batch_axis: int,
                 ndim: int) -> None:

        self._channel_axis = channel_axis
        self._quantile_axis = quantile_axis
        self._cluster_axis = cluster_axis
        self._batch_axis = batch_axis
        self._ndim = ndim

    def _create_indices(self,
                        channel_idx: Optional[int] = None,
                        quantile_idx: Optional[int] = None,
                        cluster_idx: Optional[int] = None,
                        batch_idx: Optional[int] = None) -> tuple[slice, ...]:
        """\
        returns a tuple of slice objects to get the correct insertion site
        """
        slices = [slice(None) for _ in range(self._ndim)]
        if channel_idx is not None:
            slices[self._channel_axis] = slice(channel_idx,
                                               channel_idx + 1)
        if quantile_idx is not None:
            slices[self._quantile_axis] = slice(quantile_idx,
                                                quantile_idx + 1)
        if cluster_idx is not None:
            slices[self._cluster_axis] = slice(cluster_idx,
                                               cluster_idx + 1)
        if batch_idx is not None:
            slices[self._batch_axis] = slice(batch_idx,
                                             batch_idx + 1)

        return tuple(slices)


class ExpressionQuantiles(BaseQuantileHandler):
    """\
    Calculates and holds the expression quantiles.
    """

    def __init__(self,
                 n_batches: int,
                 n_channels: int,
                 n_quantiles: int,
                 n_clusters: int,
                 quantile_array: Optional[Union[list[int], np.ndarray]] = None):

        super().__init__(
            quantile_axis = 0,
            channel_axis = 1,
            cluster_axis = 2,
            batch_axis = 3,
            ndim = 4
        )

        if quantile_array is not None:
            if not isinstance(quantile_array, np.ndarray):
                quantile_array = np.array(quantile_array)
            self._n_quantiles = quantile_array.shape[0]
            self.quantiles = quantile_array
        else:
            self._n_quantiles = n_quantiles
            self.quantiles = self._create_quantile_array()

        self._n_batches = n_batches
        self._n_channels = n_channels
        self._n_clusters = n_clusters

        self._init_array()

    def _create_quantile_array(self) -> np.ndarray:
        """
        Creates an array with the different quantiles.
        Skips 0 and 1 on purpose. Follows the R implementation
        closely, giving the exact same values.

        Alternative, including 0 and 1:
            return np.linspace(0, 100, self._n_quantiles) / 100
        """
        # return np.linspace(0, 100, self._n_quantiles) / 100
        return (np.arange(1, self._n_quantiles + 1) / (self._n_quantiles + 1))

    def _init_array(self):
        """
        initializes numpy array to insert into. We create this
        en-bloc to avoid costly concatenation operations
        afterwards.
        """
        shape: list[int] = [0, 0, 0, 0]

        shape[self._cluster_axis] = self._n_clusters
        shape[self._batch_axis] = self._n_batches
        shape[self._quantile_axis] = self._n_quantiles
        shape[self._channel_axis] = self._n_channels

        self._expr_quantiles = np.zeros(
            shape = tuple(shape)
        )

    def calculate_quantiles(self,
                            data: np.ndarray) -> np.ndarray:
        """\
        Public method to calculate quantiles. The number of
        quantiles has been set during instantiation of the
        object.

        Parameters
        ----------
        data
            Numpy array of the data of which the quantiles
            are calculated

        Returns
        -------
        A :class:`np.ndarray` containing the quantiles along axis 0 of the input data.

        """
        return self._calculate_quantiles(data)

    def _calculate_quantiles(self,
                             data: np.ndarray) -> np.ndarray:
        """Calculates the quantiles from the data"""
        q = numba_quantiles(data, self.quantiles)
        # q = np.quantile(data, self.quantiles, axis = 0)

        # alternative return syntax: q[(...,) + (np.newaxis,) * self._expr_quantiles.ndim]
        # needs testing... not sure if more readable but surely more generic
        return q[:, :, np.newaxis, np.newaxis]

    def calculate_and_add_quantiles(self,
                                    data: np.ndarray,
                                    batch_idx: int,
                                    cluster_idx: int) -> None:
        """\
        Calculates and adds the quantile array.

        Parameters
        ----------
        data
            Numpy array of the data of which the quantiles
            are calculated
        batch_idx
            The batch axis that is used for inserting the values
        cluster_idx
            The cluster axis that is used for inserting the values

        Returns
        -------
        None

        """
        quantile_array = self.calculate_quantiles(data)
        self.add_quantiles(quantile_array, batch_idx, cluster_idx)

    def add_quantiles(self,
                      quantile_array: np.ndarray,
                      batch_idx: int,
                      cluster_idx: int) -> None:
        """\
        Adds quantile arrays of shape n_channels x n_quantile.

        Parameters
        ----------
        quantile_array
            Numpy array of shape n_channels x n_quantiles.
        batch_idx
            The batch axis that is used for inserting the values
        cluster_idx
            The cluster axis that is used for inserting the values

        Returns
        -------
        None

        """

        self._expr_quantiles[
            self._create_indices(cluster_idx = cluster_idx,
                                 batch_idx = batch_idx)
        ] = quantile_array

    def add_nan_slice(self,
                      batch_idx: int,
                      cluster_idx: int) -> None:
        """\
        Adds np.nan of shape n_channels x n_quantile. This is needed
        if there are no cells in a specific cluster.

        Parameters
        ----------
        quantile_array
            Numpy array of shape n_channels x n_quantiles.
        batch_idx
            The batch axis that is used for inserting the values
        cluster_idx
            The cluster axis that is used for inserting the values

        Returns
        -------
        None

        """
        eq_shape = list(self._expr_quantiles.shape)
        arr = np.empty(
            shape = (
                eq_shape[self._quantile_axis],
                eq_shape[self._channel_axis]
            )
        )
        arr[:] = np.nan
        arr = arr[:, :, np.newaxis, np.newaxis]
        self.add_quantiles(arr, batch_idx, cluster_idx)

    def _is_nan_slice(self,
                      data) -> np.bool_:
        return np.all(np.isnan(data))

    def get_quantiles(self,
                      channel_idx: Optional[int] = None,
                      quantile_idx: Optional[int] = None,
                      cluster_idx: Optional[int] = None,
                      batch_idx: Optional[int] = None,
                      flattened: bool = True) -> np.ndarray:
        """\
        Returns a quantile array.

        channel_idx
            The index of the channel to be returned.
        quantile_idx
            The index of the quantile to be returned.
        cluster_idx
            The index of the cluster to be returned.
        batch_idx
            The index of the batch to be returned.
        flattened
            If `True`, a flattened array is returned.

        Returns
        -------
        A :class:`np.ndarray` containing the expression values.

        """
        idxs = self._create_indices(channel_idx = channel_idx,
                                    quantile_idx = quantile_idx,
                                    cluster_idx = cluster_idx,
                                    batch_idx = batch_idx)
        q = self._expr_quantiles[idxs]
        if flattened:
            return q.flatten()
        return q


class GoalDistribution(BaseQuantileHandler):
    """\
    Calculates and stores the goal distribution of the expression
    values.

    Parameters
    ----------
    expr_quantiles
        The expression values divided by batch, quantiles and clusters.
    goal
        Function to calculate the goal expression per cluster and
        channel. Defaults to `batch_mean`, which calculates the mean
        over all batches. Can be any of `batch_median`, `batch_mean`
        or a specific batch passed as an int.

    Returns
    -------
    None
        
    """

    def __init__(self,
                 expr_quantiles: ExpressionQuantiles,
                 goal: Union[int, str] = "batch_mean"):

        super().__init__(
            quantile_axis = expr_quantiles._quantile_axis,
            channel_axis = expr_quantiles._channel_axis,
            cluster_axis = expr_quantiles._cluster_axis,
            batch_axis = expr_quantiles._batch_axis,
            ndim = expr_quantiles._ndim
        )

        if goal == "batch_mean":
            if np.isnan(expr_quantiles._expr_quantiles).any():
                mean_func: Callable = np.nanmean
            else:
                mean_func: Callable = np.mean
            self.distrib = mean_func(
                expr_quantiles._expr_quantiles,
                axis = self._batch_axis
            )
            self.distrib = self.distrib[:, :, :, np.newaxis]
        elif goal == "batch_median":
            if np.isnan(expr_quantiles._expr_quantiles).any():
                mean_func: Callable = np.nanmedian
            else:
                mean_func: Callable = np.median
            self.distrib = mean_func(
                expr_quantiles._expr_quantiles,
                axis = self._batch_axis
            )
            self.distrib = self.distrib[:, :, :, np.newaxis]
        else:
            assert isinstance(goal, int)
            self.distrib = expr_quantiles.get_quantiles(batch_idx = goal)

    def get_quantiles(self,
                      channel_idx: Optional[int],
                      quantile_idx: Optional[int],
                      cluster_idx: Optional[int],
                      batch_idx: Optional[int],
                      flattened: bool = True) -> np.ndarray:
        """\
        Returns a quantile array.

        channel_idx
            The index of the channel to be returned.
        quantile_idx
            The index of the quantile to be returned.
        cluster_idx
            The index of the cluster to be returned.
        batch_idx
            The index of the batch to be returned.
        flattened
            If `True`, a flattened array is returned.

        Returns
        -------
        A :class:`np.ndarray` containing the expression values.

        """
        idxs = self._create_indices(channel_idx = channel_idx,
                                    quantile_idx = quantile_idx,
                                    cluster_idx = cluster_idx,
                                    batch_idx = batch_idx)
        d = self.distrib[idxs]
        if flattened:
            return d.flatten()
        return d
