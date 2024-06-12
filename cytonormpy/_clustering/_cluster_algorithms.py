import numpy as np

from typing import Optional

from flowsom.models import FlowSOMEstimator
from sklearn.cluster import KMeans as knnclassifier
from sklearn.cluster import AffinityPropagation as affinitypropagationclassifier
from sklearn.cluster import MeanShift as meanshiftclassifier

from abc import abstractmethod


class ClusterBase:
    """\
    Template for a clustering object. In the future, more
    clustering algorithms will be implemented.
    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self,
              X: np.ndarray,
              **kwargs) -> None:
        pass

    @abstractmethod
    def calculate_clusters(self,
                           X: np.ndarray,
                           **kwargs) -> np.ndarray:
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

    def __init__(self,
                 **kwargs):
        super().__init__()
        if not kwargs:
            kwargs = {}
        if "n_clusters" not in kwargs:
            kwargs["n_clusters"] = 30
        if "seed" not in kwargs:
            kwargs["seed"] = 187
        self.est = FlowSOMEstimator(**kwargs)

    def train(self,
              X: np.ndarray,
              **kwargs):
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

    def calculate_clusters(self,
                           X: np.ndarray,
                           **kwargs) -> np.ndarray:
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

    def __init__(self,
                 **kwargs):
        super().__init__()
        if "random_state" not in kwargs:
            kwargs["random_state"] = 187
        self.est = meanshiftclassifier(**kwargs)

    def train(self,
              X: np.ndarray,
              **kwargs):
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

    def calculate_clusters(self,
                           X: np.ndarray,
                           **kwargs) -> np.ndarray:
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

    def __init__(self,
                 **kwargs):
        super().__init__()
        if "random_state" not in kwargs:
            kwargs["random_state"] = 187
        self.est = knnclassifier(**kwargs)

    def train(self,
              X: np.ndarray,
              **kwargs):
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

    def calculate_clusters(self,
                           X: np.ndarray,
                           **kwargs) -> np.ndarray:
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

    def __init__(self,
                 **kwargs):
        super().__init__()
        if "random_state" not in kwargs:
            kwargs["random_state"] = 187
        self.est = affinitypropagationclassifier(**kwargs)

    def train(self,
              X: np.ndarray,
              **kwargs):
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

    def calculate_clusters(self,
                           X: np.ndarray,
                           **kwargs) -> np.ndarray:
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
