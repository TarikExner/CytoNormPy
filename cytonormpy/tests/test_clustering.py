import pytest
from anndata import AnnData
from pathlib import Path
import numpy as np
import pandas as pd
from cytonormpy import CytoNorm
import cytonormpy as cnp
from cytonormpy._transformation._transformations import AsinhTransformer
from cytonormpy._clustering._cluster_algorithms import (
    FlowSOM,
    ClusterBase,
    KMeans,
    AffinityPropagation,
    MeanShift,
)
from cytonormpy._cytonorm._utils import ClusterCVWarning, _calculate_cluster_cv

from sklearn.cluster import MeanShift as SM_MeanShift
from sklearn.cluster import AffinityPropagation as SM_AffinityPropagation
from sklearn.cluster import KMeans as SK_KMeans


class DummyDataHandler:
    """A fake datahandler that returns a DataFrame with a sample_key in its index."""

    def __init__(self, df: pd.DataFrame, sample_key: str):
        self._df = df
        self.metadata = type("M", (), {"sample_identifier_column": sample_key})

    def get_ref_data_df(self, markers=None):
        return self._df.copy()

    def get_ref_data_df_subsampled(self, markers=None, n=None):
        return self._df.copy()


class DummyClusterer:
    """A fake clusterer with a calculate_clusters_multiple method."""

    def __init__(self, assignments: np.ndarray):
        """
        assignments: shape (n_cells, n_tests)
        """
        self._assign = assignments

    def calculate_clusters_multiple(self, *args, **kwargs):
        # ignore X, just return the prebuilt array
        return self._assign


def test_run_clustering(data_anndata: AnnData):
    cn = CytoNorm()
    cn.run_anndata_setup(adata=data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(FlowSOM())
    cn.run_clustering(n_cells=100, test_cluster_cv=False, cluster_cv_threshold=2)
    assert "clusters" in cn._datahandler.ref_data_df.index.names


def test_run_clustering_appropriate_clustering(data_anndata: AnnData):
    cn = CytoNorm()
    cn.run_anndata_setup(adata=data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(FlowSOM())
    cn.run_clustering(n_cells=100, test_cluster_cv=True, cluster_cv_threshold=2)
    assert "clusters" in cn._datahandler.ref_data_df.index.names


def test_run_clustering_above_cv(metadata: pd.DataFrame, INPUT_DIR: Path):
    cn = cnp.CytoNorm()
    # cn.run_anndata_setup(adata = data_anndata)
    fs = FlowSOM(n_jobs=1, metacluster_kwargs={"L": 14, "K": 15})
    assert isinstance(fs, FlowSOM)
    assert isinstance(fs, ClusterBase)
    cn.add_clusterer(fs)
    t = AsinhTransformer()
    cn.add_transformer(t)
    cn.run_fcs_data_setup(metadata=metadata, input_directory=INPUT_DIR, channels="markers")
    with pytest.warns(ClusterCVWarning, match="above the threshold."):
        cn.run_clustering(cluster_cv_threshold=0)
    assert "clusters" in cn._datahandler.ref_data_df.index.names


def test_run_clustering_with_markers(data_anndata: AnnData, detector_subset: list[str]):
    cn = CytoNorm()
    cn.run_anndata_setup(adata=data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(FlowSOM())
    ref_data_df = cn._datahandler.ref_data_df
    original_shape = ref_data_df.shape
    cn.run_clustering(
        n_cells=100, test_cluster_cv=True, cluster_cv_threshold=2, markers=detector_subset
    )
    assert "clusters" in cn._datahandler.ref_data_df.index.names
    assert cn._datahandler.ref_data_df.shape == original_shape
    # we check if the rest works
    cn.calculate_quantiles()
    cn.calculate_splines()
    cn.normalize_data()


def test_wrong_input_shape_for_clustering(data_anndata: AnnData, detector_subset: list[str]):
    cn = CytoNorm()
    cn.run_anndata_setup(adata=data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(FlowSOM())
    flowsom = cn._clustering
    assert flowsom is not None

    train_data_df = cn._datahandler.get_ref_data_df(markers=detector_subset)
    assert train_data_df.shape[1] == len(detector_subset)
    train_array = train_data_df.to_numpy(copy=True)
    assert train_array.shape[1] == len(detector_subset)
    flowsom.train(X=train_array)

    # we deliberately get the full dataframe
    ref_data_df = cn._datahandler.get_ref_data_df(markers=None).copy()
    assert ref_data_df.shape[1] != len(detector_subset)
    subset_ref_data_df = cn._datahandler.get_ref_data_df(markers=detector_subset).copy()
    assert subset_ref_data_df.shape[1] == len(detector_subset)

    # this shouldn't be possible since we train and predict on different shapes...
    predict_array_large = ref_data_df.to_numpy(copy=True)
    assert predict_array_large.shape[1] != len(detector_subset)
    with pytest.raises(ValueError):
        flowsom.calculate_clusters(X=predict_array_large)


def test_wrong_input_shape_for_clustering_kmeans(data_anndata: AnnData, detector_subset: list[str]):
    cn = CytoNorm()
    cn.run_anndata_setup(adata=data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(KMeans())
    flowsom = cn._clustering
    train_data_df = cn._datahandler.get_ref_data_df(markers=detector_subset)
    assert train_data_df.shape[1] == len(detector_subset)
    train_array = train_data_df.to_numpy(copy=True)
    assert train_array.shape[1] == len(detector_subset)
    flowsom.train(X=train_array)

    # we deliberately get the full dataframe
    ref_data_df = cn._datahandler.get_ref_data_df(markers=None).copy()
    assert ref_data_df.shape[1] != len(detector_subset)
    subset_ref_data_df = cn._datahandler.get_ref_data_df(markers=detector_subset).copy()
    assert subset_ref_data_df.shape[1] == len(detector_subset)

    # this shouldn't be possible since we train and predict on different shapes...
    predict_array_large = ref_data_df.to_numpy(copy=True)
    assert predict_array_large.shape[1] != len(detector_subset)
    with pytest.raises(ValueError):
        flowsom.calculate_clusters(X=predict_array_large)


def make_indexed_df(sample_ids: list[str], n_cells: int) -> pd.DataFrame:
    """
    Build a DataFrame with a MultiIndex on 'sample_id' for n_cells,
    evenly split across those sample_ids.
    """
    repeats = n_cells // len(sample_ids)
    idx = []
    for s in sample_ids:
        idx += [s] * repeats
    # if n_cells not divisible, pad with first sample
    idx += [sample_ids[0]] * (n_cells - len(idx))
    return pd.DataFrame(
        data=np.zeros((n_cells, 1)), index=pd.Index(idx, name="file"), columns=["dummy"]
    )


def test_calculate_cluster_cvs_structure(monkeypatch):
    # Create a fake CytoNorm
    cn = CytoNorm()
    # Dummy data: 6 cells, 3 for 'A', 3 for 'B'
    df = make_indexed_df(["A", "B"], n_cells=6)
    cn._datahandler = DummyDataHandler(df, sample_key="file")

    # Suppose we test k=1 and k=2, and we want assignments shaped (6,2)
    # For k=1 all cells in cluster 0; for k=2, first 3 cells→0, last 3→1
    assign = np.vstack(
        [np.zeros(6, int), np.concatenate([np.zeros(3, int), np.ones(3, int)])]
    ).T  # shape (6,2)
    cn._clustering = DummyClusterer(assign)

    _ = cn.calculate_cluster_cvs([1, 2])  # returns None but sets cn.cvs_by_k
    assert isinstance(cn.cvs_by_k, dict)

    # keys must match requested k’s
    assert set(cn.cvs_by_k.keys()) == {1, 2}
    # for k=1, list length 1; for k=2, length 2
    assert len(cn.cvs_by_k[1]) == 1
    assert len(cn.cvs_by_k[2]) == 2

    # each entry should be a float
    for vs in cn.cvs_by_k.values():
        assert all(isinstance(x, float) for x in vs)


def test_calculate_cluster_cv_values():
    # Build a tiny DataFrame with 4 cells and 2 samples
    # sample X has two cells in cluster 0; sample Y has two cells in cluster 1
    df = pd.DataFrame({"file": ["X", "X", "Y", "Y"], "cluster": [0, 0, 1, 1]})
    # cluster 0: proportions across samples = [2/2, 0/2] = [1,0]
    #   mean=0.5, sd=0.7071 → CV≈1.4142
    # cluster 1: [0,1] → same CV
    cvs = _calculate_cluster_cv(df, cluster_key="cluster", sample_key="file")
    # verify pivot table size and values
    # check CVs
    expected_cv = np.std([1, 0], ddof=1) / np.mean([1, 0])
    assert pytest.approx(expected_cv, rel=1e-3) == cvs[0]
    assert pytest.approx(expected_cv, rel=1e-3) == cvs[1]


@pytest.fixture
def toy_data():
    # simple 1D clusters: [0,0,0, 1,1,1]
    return np.array([[i] for i in [0, 0, 0, 5, 5, 5]])


def test_mean_shift_multiple_warnings_and_identity(toy_data):
    ms = MeanShift(bandwidth=2.0)  # any bandwidth
    # monkey‑patch underlying sklearn estimator so fit/predict work
    ms.est = SM_MeanShift(bandwidth=2.0)
    # ask for 3 different k’s
    ks = [2, 3, 5]
    with pytest.warns(UserWarning) as record:
        out = ms.calculate_clusters_multiple(toy_data, ks)
    # exactly one warning
    assert len(record) == 1
    assert "MeanShift: ignoring requested n_clusters" in str(record[0].message)
    # output shape
    assert out.shape == (6, 3)
    # all columns identical
    assert np.all(out[:, 0] == out[:, 1]) and np.all(out[:, 1] == out[:, 2])


def test_affinity_propagation_multiple_warnings_and_identity(toy_data):
    ap = AffinityPropagation(damping=0.9)
    ap.est = SM_AffinityPropagation(damping=0.9)
    ks = [1, 2]
    with pytest.warns(UserWarning) as record:
        out = ap.calculate_clusters_multiple(toy_data, ks)
    assert "AffinityPropagation: ignoring requested n_clusters" in str(record[0].message)
    assert out.shape == (6, 2)
    assert np.all(out[:, 0] == out[:, 1])


def test_kmeans_multiple_varies_clusters(toy_data):
    km = KMeans(n_clusters=2, random_state=42)
    km.est = SK_KMeans(n_clusters=2, random_state=42)
    ks = [2, 3, 4]
    out = km.calculate_clusters_multiple(toy_data, ks)
    # no warnings
    # shape correct
    assert out.shape == (6, 3)
    diffs = [not np.array_equal(out[:, i], out[:, j]) for i in range(3) for j in range(i + 1, 3)]
    assert not any(diffs)
