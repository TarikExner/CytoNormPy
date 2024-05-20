import pytest
from anndata import AnnData
from pathlib import Path
import pandas as pd
from cytonormpy import CytoNorm
import cytonormpy as cnp
from cytonormpy._transformation._transformations import AsinhTransformer, Transformer, HyperLogTransformer
from cytonormpy._clustering._flowsom import FlowSOM, ClusterBase
from cytonormpy._dataset._dataset import DataHandlerFCS, DataHandlerAnnData
from cytonormpy._cytonorm._cytonorm import ClusterCVWarning


def test_instantiation_fcs(tmp_path: Path,
                           metadata: pd.DataFrame,
                           INPUT_DIR: Path):
    cn = CytoNorm()
    cn.run_fcs_data_setup(metadata = metadata,
                          input_directory = INPUT_DIR,
                          output_directory = tmp_path)

    assert hasattr(cn, "_datahandler")
    assert isinstance(cn._datahandler, DataHandlerFCS)


def test_instantiation_anndata(data_anndata: AnnData):
    cn = CytoNorm()
    cn.run_anndata_setup(adata = data_anndata)
    assert hasattr(cn, "_datahandler")
    assert isinstance(cn._datahandler, DataHandlerAnnData)
    assert "cyto_normalized" in cn._datahandler.adata.layers


def test_transformer_addition():
    cn = CytoNorm()
    transformer = AsinhTransformer()
    cn.add_transformer(transformer)
    assert hasattr(cn, "_transformer")
    assert isinstance(cn._transformer, AsinhTransformer)
    assert isinstance(cn._transformer, Transformer)


def test_clusterer_addition():
    cn = CytoNorm()
    clusterer = FlowSOM()
    cn.add_clusterer(clusterer)
    assert hasattr(cn, "_clustering")
    assert isinstance(cn._clustering, FlowSOM)
    assert isinstance(cn._clustering, ClusterBase)
    assert not hasattr(cn, "_transformer")


def test_run_clustering(data_anndata: AnnData):
    cn = CytoNorm()
    cn.run_anndata_setup(adata = data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(FlowSOM())
    cn.run_clustering(n_cells = 100,
                      test_cluster_cv = False,
                      cluster_cv_threshold = 2)
    assert "clusters" in cn._datahandler.ref_data_df.index.names


def test_run_clustering_appropriate_clustering(data_anndata: AnnData):
    cn = CytoNorm()
    cn.run_anndata_setup(adata = data_anndata)
    cn.add_transformer(AsinhTransformer())
    cn.add_clusterer(FlowSOM())
    cn.run_clustering(n_cells = 100,
                      test_cluster_cv = True,
                      cluster_cv_threshold = 2)
    assert "clusters" in cn._datahandler.ref_data_df.index.names


def test_run_clustering_above_cv(data_anndata: AnnData,
                                 metadata: pd.DataFrame,
                                 INPUT_DIR: Path):
    cn = cnp.CytoNorm()
    # cn.run_anndata_setup(adata = data_anndata)
    fs = FlowSOM(n_jobs = 1, metacluster_kwargs = {"L": 14, "K": 15})
    assert isinstance(fs, FlowSOM)
    assert isinstance(fs, ClusterBase)
    cn.add_clusterer(fs)
    t = AsinhTransformer()
    cn.add_transformer(t)
    cn.run_fcs_data_setup(metadata = metadata,
                          input_directory = INPUT_DIR,
                          channels = "markers")
    with pytest.warns(ClusterCVWarning, match = "above the threshold."):
        cn.run_clustering(cluster_cv_threshold = 0)
    assert "clusters" in cn._datahandler.ref_data_df.index.names


