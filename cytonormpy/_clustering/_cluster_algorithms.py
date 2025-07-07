import numpy as np
import warnings

from abc import abstractmethod
from flowsom.models import FlowSOMEstimator
from sklearn.base import clone
from sklearn.cluster import KMeans as knnclassifier
from sklearn.cluster import AffinityPropagation as affinitypropagationclassifier
from sklearn.cluster import MeanShift as meanshiftclassifier


class ClusterBase:
    """\
    Template for a clustering object. In the future, more
    clustering algorithms will be implemented.
    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self, X: np.ndarray, **kwargs) -> None:
        pass

    @abstractmethod
    def calculate_clusters(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_clusters_multiple(self, X: np.ndarray, n_clusters: list[int]) -> np.ndarray:
        pass


class FlowSOM(ClusterBase):
    """\
    Class to perform FlowSOM clustering.

    Parameters
    ----------
    kwargs
        keyword arguments passed to :class:`flowsom.FlowSOMEstimator`.
        For further information, refer to their documentation.

    Returns
    -------
    None

    """

    def __init__(self, **kwargs):
        super().__init__()
        if not kwargs:
            kwargs = {}
        if "n_clusters" not in kwargs:
            kwargs["n_clusters"] = 30
        if "seed" not in kwargs:
            kwargs["seed"] = 187
        self.est = FlowSOMEstimator(**kwargs)

    def train(self, X: np.ndarray, **kwargs):
        """\
        Trains the SOM. Calls :class:`flowsom.FlowSOMEstimator.fit()` internally.

        Parameters
        ----------
        X
            The data used for traning.
        kwargs
            Keyword arguments passed to :class:`flowsom.FlowSOMEstimator.fit()`

        Returns
        -------
        None

        """
        self.est.fit(X, **kwargs)
        return

    def calculate_clusters(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """\
        Calculates the clusters. Calls :class:`flowsom.FlowSOMEstimator.predict()` internally.

        Parameters
        ----------
        X
            The data that are supposed to be predicted.
        kwargs
            Keyword arguments passed to :class:`flowsom.FlowSOMEstimator.predict()`

        Returns
        -------
        Cluster annotations stored in a :class:`np.ndarray`

        """
        return self.est.predict(X, **kwargs)

    def calculate_clusters_multiple(self, X: np.ndarray, n_clusters: list[int]):
        """\
        Calculates the clusters for a given metacluster number. The estimator
        will calculate a SOM once, then fit the ConsensusCluster class given
        the n_metaclusters that are provided.

        Parameters
        ----------
        X
            The data that are supposed to be predicted.
        n_metaclusters
            A list of integers specifying the number of metaclusters per test.

        Returns
        -------
        Cluster annotations stored in a :class:`np.ndarray`, where the n_metacluster
        denotes the column and the rows are the individual cells.

        """
        self.est.cluster_model.fit(X)
        y_clusters = self.est.cluster_model.predict(X)
        X_codes = self.est.cluster_model.codes
        assignments = np.empty((X.shape[0], len(n_clusters)), dtype=np.int16)
        for j, n_mc in enumerate(n_clusters):
            self.est.set_n_clusters(n_mc)
            y_codes = self.est.metacluster_model.fit_predict(X_codes)
            assignments[:, j] = y_codes[y_clusters]
        return assignments


class MeanShift(ClusterBase):
    """\
    Class to perform MeanShift clustering.

    Parameters
    ----------
    kwargs
        keyword arguments passed to :class:`sklearn.cluster.MeanShift`.
        For further information, refer to their documentation.

    Returns
    -------
    None

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.est = meanshiftclassifier(**kwargs)

    def train(self, X: np.ndarray, **kwargs):
        """\
        Trains the classifier. Calls :class:`sklearn.cluster.MeanShift.fit()` internally.

        Parameters
        ----------
        X
            The data used for traning.
        kwargs
            Keyword arguments passed to :class:`sklearn.cluster.MeanShift.fit()`

        Returns
        -------
        None

        """
        self.est.fit(X, **kwargs)
        return

    def calculate_clusters(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """\
        Calculates the clusters. Calls :class:`sklearn.cluster.MeanShift.predict()` internally.

        Parameters
        ----------
        X
            The data that are supposed to be predicted.
        kwargs
            Keyword arguments passed to :class:`sklearn.cluster.MeanShift.predict()`

        Returns
        -------
        Cluster annotations stored in a :class:`np.ndarray`

        """
        return self.est.predict(X, **kwargs)

    def calculate_clusters_multiple(self, X: np.ndarray, n_clusters: list[int]):
        """
        MeanShift ignores n_clusters: warns if len(n_clusters)>1,
        then returns the same assignment in each column.
        """
        if len(n_clusters) > 1:
            warnings.warn(
                "MeanShift: ignoring requested n_clusters list, "
                "producing identical assignments for each entry.",
                UserWarning,
                stacklevel=2,
            )

        n_samples = X.shape[0]
        out = np.empty((n_samples, len(n_clusters)), dtype=int)

        for j in range(len(n_clusters)):
            est = clone(self.est)
            est.fit(X)
            out[:, j] = est.predict(X)

        return out


class KMeans(ClusterBase):
    """\
    Class to perform KMeans clustering.

    Parameters
    ----------
    kwargs
        keyword arguments passed to :class:`sklearn.cluster.KMeans`.
        For further information, refer to their documentation.

    Returns
    -------
    None

    """

    def __init__(self, **kwargs):
        super().__init__()
        if "random_state" not in kwargs:
            kwargs["random_state"] = 187
        self.est = knnclassifier(**kwargs)

    def train(self, X: np.ndarray, **kwargs):
        """\
        Trains the classifier. Calls :class:`sklearn.cluster.KMeans.fit()` internally.

        Parameters
        ----------
        X
            The data used for traning.
        kwargs
            Keyword arguments passed to :class:`sklearn.cluster.KMeans.fit()`

        Returns
        -------
        None

        """
        self.est.fit(X, **kwargs)
        return

    def calculate_clusters(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """\
        Calculates the clusters. Calls :class:`sklearn.cluster.KMeans.predict()` internally.

        Parameters
        ----------
        X
            The data that are supposed to be predicted.
        kwargs
            Keyword arguments passed to :class:`sklearn.cluster.KMeans.predict()`

        Returns
        -------
        Cluster annotations stored in a :class:`np.ndarray`

        """
        return self.est.predict(X, **kwargs)

    def calculate_clusters_multiple(self, X: np.ndarray, n_clusters: list[int]):
        """
        Returns an array of shape (n_samples, len(n_clusters)),
        where each column i is the cluster‐assignment vector
        for KMeans(n_clusters=n_clusters[i]).
        """
        n_samples = X.shape[0]
        out = np.empty((n_samples, len(n_clusters)), dtype=int)

        for j, k in enumerate(n_clusters):
            est = clone(self.est)
            est.set_params(n_clusters=k)
            est.fit(X)
            out[:, j] = est.predict(X)

        return out


class AffinityPropagation(ClusterBase):
    """\
    Class to perform AffinityPropagation clustering.

    Parameters
    ----------
    kwargs
        keyword arguments passed to :class:`sklearn.cluster.AffinityPropagation`.
        For further information, refer to their documentation.

    Returns
    -------
    None

    """

    def __init__(self, **kwargs):
        super().__init__()
        if "random_state" not in kwargs:
            kwargs["random_state"] = 187
        self.est = affinitypropagationclassifier(**kwargs)

    def train(self, X: np.ndarray, **kwargs):
        """\
        Trains the classifier. Calls :class:`sklearn.cluster.AffinityPropagation.fit()` internally.

        Parameters
        ----------
        X
            The data used for traning.
        kwargs
            Keyword arguments passed to :class:`sklearn.cluster.AffinityPropagation.fit()`

        Returns
        -------
        None

        """
        self.est.fit(X, **kwargs)
        return

    def calculate_clusters(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """\
        Calculates the clusters. Calls :class:`sklearn.cluster.AffinityPropagation.predict()` internally.

        Parameters
        ----------
        X
            The data that are supposed to be predicted.
        kwargs
            Keyword arguments passed to :class:`sklearn.cluster.AffinityPropagation.predict()`

        Returns
        -------
        Cluster annotations stored in a :class:`np.ndarray`

        """
        return self.est.predict(X, **kwargs)

    def calculate_clusters_multiple(self, X: np.ndarray, n_clusters: list[int]):
        """
        AffinityPropagation ignores n_clusters: warns if len(n_clusters)>1,
        then returns the same assignment for each entry.
        """
        if len(n_clusters) > 1:
            warnings.warn(
                "AffinityPropagation: ignoring requested n_clusters list, "
                "producing identical assignments for each entry.",
                UserWarning,
                stacklevel=2,
            )

        n_samples = X.shape[0]
        out = np.empty((n_samples, len(n_clusters)), dtype=int)

        for j in range(len(n_clusters)):
            est = clone(self.est)
            est.fit(X)
            out[:, j] = est.predict(X)

        return out
