import pytest
from anndata import AnnData
from pathlib import Path
import pandas as pd
from cytonormpy import CytoNorm
import cytonormpy as cnp
from cytonormpy._transformation._transformations import AsinhTransformer
from cytonormpy._clustering._cluster_algorithms import FlowSOM, ClusterBase, KMeans
from cytonormpy._cytonorm._utils import ClusterCVWarning


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


def test_wrong_input_shape_for_clustering(data_anndata: AnnData, detector_subset: list[str]):
    cn = CytoNorm()
    cn.run_anndata_setup(adata=data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(FlowSOM())
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
