import pytest
import anndata as ad
import os
from anndata import AnnData
from pathlib import Path
import pandas as pd
import numpy as np
from cytonormpy import CytoNorm, FCSFile
import cytonormpy as cnp
import warnings
from cytonormpy._transformation._transformations import AsinhTransformer, Transformer
from cytonormpy._clustering._cluster_algorithms import FlowSOM, ClusterBase
from cytonormpy._dataset._dataset import DataHandlerFCS, DataHandlerAnnData
from cytonormpy._cytonorm._cytonorm import ClusterCVWarning
from cytonormpy._normalization._quantile_calc import ExpressionQuantiles


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
    assert hasattr(cn, "_transformer")
    assert cn._transformer is None


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


def test_run_clustering_above_cv(metadata: pd.DataFrame,
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

def test_fancy_numpy_indexing_without_clustering(metadata: pd.DataFrame,
                                                 INPUT_DIR: Path):
    cn = cnp.CytoNorm()
    t = cnp.AsinhTransformer()
    cn.add_transformer(t)
    cn.run_fcs_data_setup(input_directory = INPUT_DIR,
                          metadata = metadata,
                          channels = "markers",
                          output_directory = INPUT_DIR)
    
    # we compare the df.loc with our numpy indexing
    ref_data_df: pd.DataFrame = cn._datahandler.get_ref_data_df()
    if "clusters" not in ref_data_df.index.names:
        ref_data_df["clusters"] = -1
        ref_data_df.set_index("clusters", append = True, inplace = True)
    
    ref_data_df = ref_data_df.sort_index()

    # we extract the values for batch and cluster
    # and get the idxs of their unique combinations
    batch_idxs = ref_data_df.index.get_level_values("batch").to_numpy()
    cluster_idxs = ref_data_df.index.get_level_values("clusters").to_numpy()
    batch_cluster_idxs = np.vstack([batch_idxs, cluster_idxs]).T
    batch_cluster_unique_idxs = np.unique(
        batch_cluster_idxs,
        axis = 0,
        return_index = True
    )[1]
    # we append the shape as last idx
    batch_cluster_unique_idxs = np.hstack(
        [
            batch_cluster_unique_idxs,
            np.array(
                batch_cluster_idxs.shape[0]
            )
        ]
    )

    # we create a lookup table to get the batch and cluster back
    batch_cluster_lookup = {
        idx: [batch_idxs[idx], cluster_idxs[idx]]
        for idx in batch_cluster_unique_idxs[:-1]
    }

    ref_data = ref_data_df.to_numpy()

    for i in range(batch_cluster_unique_idxs.shape[0]-1):
        batch, cluster = batch_cluster_lookup[batch_cluster_unique_idxs[i]]
        data = ref_data[
            batch_cluster_unique_idxs[i] : batch_cluster_unique_idxs[i+1],
            :
        ]

        conventional_lookup = ref_data_df.loc[
            (ref_data_df.index.get_level_values("batch") == batch) &
            (ref_data_df.index.get_level_values("clusters") == cluster),
            :
        ].to_numpy()

        assert np.array_equal(data, conventional_lookup)


def test_fancy_numpy_indexing_with_clustering(metadata: pd.DataFrame,
                                              INPUT_DIR: Path):
    cn = cnp.CytoNorm()
    t = cnp.AsinhTransformer()
    cn.add_transformer(t)
    fs = FlowSOM(n_clusters = 10, xdim = 5, ydim = 5)
    cn.add_clusterer(fs)
    cn.run_fcs_data_setup(input_directory = INPUT_DIR,
                          metadata = metadata,
                          channels = "markers",
                          output_directory = INPUT_DIR)
    cn.run_clustering()
    
    # we compare the df.loc with our numpy indexing
    ref_data_df: pd.DataFrame = cn._datahandler.get_ref_data_df()
    
    ref_data_df = ref_data_df.sort_index()

    # we extract the values for batch and cluster
    # and get the idxs of their unique combinations
    batch_idxs = ref_data_df.index.get_level_values("batch").to_numpy()
    cluster_idxs = ref_data_df.index.get_level_values("clusters").to_numpy()
    batch_cluster_idxs = np.vstack([batch_idxs, cluster_idxs]).T
    batch_cluster_unique_idxs = np.unique(
        batch_cluster_idxs,
        axis = 0,
        return_index = True
    )[1]
    # we append the shape as last idx
    batch_cluster_unique_idxs = np.hstack(
        [
            batch_cluster_unique_idxs,
            np.array(
                batch_cluster_idxs.shape[0]
            )
        ]
    )

    # we create a lookup table to get the batch and cluster back
    batch_cluster_lookup = {
        idx: [batch_idxs[idx], cluster_idxs[idx]]
        for idx in batch_cluster_unique_idxs[:-1]
    }

    ref_data = ref_data_df.to_numpy()

    for i in range(batch_cluster_unique_idxs.shape[0]-1):
        batch, cluster = batch_cluster_lookup[batch_cluster_unique_idxs[i]]
        data = ref_data[
            batch_cluster_unique_idxs[i] : batch_cluster_unique_idxs[i+1],
            :
        ]

        conventional_lookup = ref_data_df.loc[
            (ref_data_df.index.get_level_values("batch") == batch) &
            (ref_data_df.index.get_level_values("clusters") == cluster),
            :
        ].to_numpy()

        assert np.array_equal(data, conventional_lookup)


def test_fancy_numpy_indexing_with_clustering_batch_cluster_idxs(metadata: pd.DataFrame,
                                                                 INPUT_DIR: Path):
    cn = cnp.CytoNorm()
    t = cnp.AsinhTransformer()
    cn.add_transformer(t)
    fs = FlowSOM(n_clusters = 10, xdim = 5, ydim = 5)
    cn.add_clusterer(fs)
    cn.run_fcs_data_setup(input_directory = INPUT_DIR,
                          metadata = metadata,
                          channels = "markers",
                          output_directory = INPUT_DIR)
    cn.run_clustering()
    
    # we compare the df.loc with our numpy indexing
    ref_data_df: pd.DataFrame = cn._datahandler.get_ref_data_df()
    
    ref_data_df = ref_data_df.sort_index()

    # we extract the values for batch and cluster
    # and get the idxs of their unique combinations
    batch_idxs = ref_data_df.index.get_level_values("batch").to_numpy()
    cluster_idxs = ref_data_df.index.get_level_values("clusters").to_numpy()
    batch_cluster_idxs = np.vstack([batch_idxs, cluster_idxs]).T
    unique_combinations, batch_cluster_unique_idxs = np.unique(
        batch_cluster_idxs,
        axis = 0,
        return_index = True
    )
    # we append the shape as last idx
    batch_cluster_unique_idxs = np.hstack(
        [
            batch_cluster_unique_idxs,
            np.array(
                batch_cluster_idxs.shape[0]
            )
        ]
    )

    # we create a lookup table to get the batch and cluster back
    batch_cluster_lookup = {
        idx: unique_combinations[i]
        for i, idx in enumerate(batch_cluster_unique_idxs[:-1])
    }
    batches = sorted(
        ref_data_df.index \
        .get_level_values("batch") \
        .unique() \
        .tolist()
    )
    clusters = sorted(
        ref_data_df.index \
        .get_level_values("clusters") \
        .unique() \
        .tolist()
    )
    channels = ref_data_df.columns.tolist()
    # we also create a lookup table for the batch indexing...
    batch_idx_lookup = {
        batch: i
        for i, batch in enumerate(batches)
    }
    # ... and the cluster indexing
    cluster_idx_lookup = {
        cluster: i
        for i, cluster in enumerate(clusters)
    }

    def find_i(batch, cluster, batch_cluster_lookup):
        index = [
            idx for idx in batch_cluster_lookup
            if batch_cluster_lookup[idx][0] == batch and
            batch_cluster_lookup[idx][1] == cluster
        ][0]
        return list(batch_cluster_unique_idxs).index(index)

    ref_data = ref_data_df.to_numpy()

    for b, batch in enumerate(batches):
        for c, cluster in enumerate(clusters):
            conventional_lookup = ref_data_df.loc[
                (ref_data_df.index.get_level_values("batch") == batch) &
                (ref_data_df.index.get_level_values("clusters") == cluster),
                channels
            ].to_numpy()
            i = find_i(batch, cluster, batch_cluster_lookup)
            b_numpy = batch_idx_lookup[batch]
            assert b == b_numpy, (b, b_numpy)
            c_numpy = cluster_idx_lookup[cluster]
            assert c == c_numpy, (c, c_numpy)
            data = ref_data[
                batch_cluster_unique_idxs[i] : batch_cluster_unique_idxs[i+1],
                :
            ]

            assert np.array_equal(conventional_lookup, data)
            cn.calculate_quantiles()

            cn._expr_quantiles.calculate_and_add_quantiles(
                data = conventional_lookup,
                batch_idx = b,
                cluster_idx = c
            )
            conv_q = cn._expr_quantiles.get_quantiles(
                None,
                None,
                b,
                c
            )
            cn._expr_quantiles.calculate_and_add_quantiles(
                data = data,
                batch_idx = b,
                cluster_idx = c
            )
            numpy_q = cn._expr_quantiles.get_quantiles(
                None,
                None,
                b_numpy,
                c_numpy
            )
            assert np.array_equal(numpy_q, conv_q, equal_nan = True)



class CytoNormPandasLookupQuantileCalc(CytoNorm):
    def __init__(self):
        super().__init__()

    def calculate_quantiles(self,
                            n_quantiles: int = 99,
                            min_cells: int = 50,
                            ) -> None:

        ref_data_df: pd.DataFrame = self._datahandler.get_ref_data_df()

        if "clusters" not in ref_data_df.index.names:
            warnings.warn("No Clusters have been found.", UserWarning)
            ref_data_df["clusters"] = -1
            ref_data_df = ref_data_df.set_index("clusters", append = True)

        batches = sorted(
            ref_data_df.index \
            .get_level_values("batch") \
            .unique() \
            .tolist()
        )
        clusters = sorted(
            ref_data_df.index \
            .get_level_values("clusters") \
            .unique() \
            .tolist()
        )
        channels = ref_data_df.columns.tolist()

        self.batches = batches
        self.clusters = clusters
        self.channels = channels

        n_channels = len(ref_data_df.columns)
        n_batches = len(batches)
        n_clusters = len(clusters)

        self._expr_quantiles = ExpressionQuantiles(
            n_channels = n_channels,
            n_quantiles = n_quantiles,
            n_batches = n_batches,
            n_clusters = n_clusters
        )

        self._not_calculated = {
            batch: [] for batch in self.batches
        }
        ref_data_df = ref_data_df.sort_index()
        for b, batch in enumerate(batches):
            for c, cluster in enumerate(clusters):
                data = ref_data_df.loc[
                    (ref_data_df.index.get_level_values("batch") == batch) &
                    (ref_data_df.index.get_level_values("clusters") == cluster),
                    channels
                ].to_numpy()

                if data.shape[0] < min_cells:
                    warning_msg = f"{data.shape[0]} cells detected in batch "
                    warning_msg += f"{batch} for cluster {cluster}. "
                    warning_msg += "Skipping quantile calculation. "

                    warnings.warn(
                        warning_msg,
                        UserWarning
                    )
                    self._not_calculated[batch].append(cluster)

                    self._expr_quantiles.add_nan_slice(
                        batch_idx = b,
                        cluster_idx = c
                    )

                    continue

                self._expr_quantiles.calculate_and_add_quantiles(
                    data = data,
                    batch_idx = b,
                    cluster_idx = c
                )

        return


def test_fancy_numpy_indexing_expr_quantiles(metadata: pd.DataFrame,
                                             INPUT_DIR: Path):
    t = cnp.AsinhTransformer()
    fs = FlowSOM(n_clusters = 10, xdim = 5, ydim = 5)

    cn1 = CytoNorm()
    cn1.add_transformer(t)
    cn1.add_clusterer(fs)
    cn1.run_fcs_data_setup(input_directory = INPUT_DIR,
                           metadata = metadata,
                           channels = "markers",
                           output_directory = INPUT_DIR)
    cn1.run_clustering()
    
    cn2 = CytoNormPandasLookupQuantileCalc()
    cn2.add_transformer(t)
    cn2.add_clusterer(fs)
    cn2.run_fcs_data_setup(input_directory = INPUT_DIR,
                           metadata = metadata,
                           channels = "markers",
                           output_directory = INPUT_DIR)
    cn2.run_clustering()

    assert np.array_equal(
        cn1._datahandler.ref_data_df.to_numpy(),
        cn2._datahandler.ref_data_df.to_numpy()
    )

    cn1_df = cn1._datahandler.ref_data_df
    cn2_df = cn2._datahandler.ref_data_df
    assert np.array_equal(
        cn1_df.index.get_level_values("batch").to_numpy(),
        cn2_df.index.get_level_values("batch").to_numpy()
    )
    assert not np.array_equal(
        cn1_df.index.get_level_values("clusters").to_numpy(),
        cn2_df.index.get_level_values("clusters").to_numpy()
    )
    cn2._datahandler.ref_data_df = cn2._datahandler.ref_data_df.droplevel("clusters")
    cn2._datahandler.ref_data_df["clusters"] = cn1_df.index.get_level_values("clusters").to_numpy()
    cn2._datahandler.ref_data_df.set_index("clusters", append = True, inplace = True)

    assert (cn1._datahandler.ref_data_df.index == cn2._datahandler.ref_data_df.index).all()

    cn1.calculate_quantiles()
    cn2.calculate_quantiles()

    assert (cn1._datahandler.ref_data_df.index == cn2._datahandler.ref_data_df.index).all()
    cn1_df = cn1._datahandler.ref_data_df
    cn2_df = cn2._datahandler.ref_data_df
    assert cn1_df.equals(cn2_df)


    assert cn1._not_calculated == cn2._not_calculated

    assert cn1.batches == cn2.batches
    assert cn1.channels == cn2.channels
    assert cn1.clusters == cn2.clusters
    assert cn1._not_calculated == cn2._not_calculated

    assert np.array_equal(
        cn1._expr_quantiles._expr_quantiles,
        cn2._expr_quantiles._expr_quantiles,
        equal_nan = True
    )

def test_quantile_calc_custom_array_errors(metadata: pd.DataFrame,
                                           INPUT_DIR: Path):
    t = cnp.AsinhTransformer()

    cn = CytoNorm()
    cn.add_transformer(t)
    cn.run_fcs_data_setup(input_directory = INPUT_DIR,
                          metadata = metadata,
                          channels = "markers",
                          output_directory = INPUT_DIR)
    with pytest.raises(TypeError):
        cn.calculate_quantiles(quantile_array = pd.DataFrame())
    with pytest.raises(ValueError):
        cn.calculate_quantiles(quantile_array = [10,20,50,100])
    custom_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    custom_quantile_array = np.array(custom_quantiles)

    cn.calculate_quantiles(quantile_array = custom_quantiles)
    assert np.array_equal(cn._expr_quantiles.quantiles, custom_quantile_array)
    assert cn._expr_quantiles._n_quantiles == custom_quantile_array.shape[0]

    cn.calculate_quantiles(quantile_array = custom_quantile_array)
    assert np.array_equal(cn._expr_quantiles.quantiles, custom_quantile_array)
    assert cn._expr_quantiles._n_quantiles == custom_quantile_array.shape[0]


def test_spline_calc_limits_errors(metadata: pd.DataFrame,
                                   INPUT_DIR: Path):
    t = cnp.AsinhTransformer()

    cn = CytoNorm()
    cn.add_transformer(t)
    cn.run_fcs_data_setup(input_directory = INPUT_DIR,
                          metadata = metadata,
                          channels = "markers",
                          output_directory = INPUT_DIR)
    cn.calculate_quantiles()
    with pytest.raises(TypeError):
        cn.calculate_splines(limits = "limitless computation!")
    cn.calculate_splines(limits = [0,8])


def test_normalizing_files_that_have_been_added_later(metadata: pd.DataFrame,
                                                      INPUT_DIR: Path,
                                                      tmpdir):
    t = cnp.AsinhTransformer()

    cn = CytoNorm()
    cn.add_transformer(t)
    cn.run_fcs_data_setup(input_directory = INPUT_DIR,
                          metadata = metadata,
                          channels = "markers",
                          output_directory = tmpdir)
    cn.calculate_quantiles()
    cn.calculate_splines(limits = [0,8])
    cn.normalize_data()

    cn.normalize_data(file_names = "Gates_PTLG034_Unstim_Control_2_dup.fcs",
                      batches = 3)
    assert "Norm_Gates_PTLG034_Unstim_Control_2_dup.fcs" in os.listdir(tmpdir)
    original_fcs = FCSFile(tmpdir, "Norm_Gates_PTLG034_Unstim_Control_2.fcs")
    dup_fcs = FCSFile(tmpdir, "Norm_Gates_PTLG034_Unstim_Control_2_dup.fcs")
    assert np.array_equal(
        original_fcs.original_events,
        dup_fcs.original_events
    )

def test_normalizing_files_that_have_been_added_later_anndata(data_anndata: AnnData):
    adata = data_anndata
    file_name = "Gates_PTLG034_Unstim_Control_2.fcs"
    file_spec_adata = adata[adata.obs["file_name"] == file_name, :].copy()
    dup_filename = "Gates_PTLG034_Unstim_Control_2_dup.fcs"
    file_spec_adata.obs["file_name"] = dup_filename
    adata.obs["batch"] = adata.obs["batch"].astype(np.int8)

    cn = CytoNorm()
    cn.run_anndata_setup(adata = adata)
    cn.calculate_quantiles()
    cn.calculate_splines()
    cn.normalize_data()
    assert "cyto_normalized" in adata.layers.keys()

    longer_adata = ad.concat([adata, file_spec_adata], axis = 0, join = "outer")
    longer_adata.obs_names_make_unique()
    assert "cyto_normalized" in longer_adata.layers.keys()

    cn.normalize_data(adata = longer_adata,
                      file_names = dup_filename,
                      batches = 3)
    assert "cyto_normalized" in longer_adata.layers.keys()

    file_adata = longer_adata[longer_adata.obs["file_name"] == file_name,:].copy()
    dup_file_adata = longer_adata[longer_adata.obs["file_name"] == dup_filename,:].copy()

    assert np.array_equal(
        file_adata.layers["cyto_normalized"],
        dup_file_adata.layers["cyto_normalized"]
    )
    
def test_normalizing_files_that_have_been_added_later_valueerror():
    cn = CytoNorm()
    with pytest.raises(ValueError):
        cn.normalize_data(file_names = "Gates_PTLG034_Unstim_Control_2_dup.fcs",
                          batches = [3, 4])


def test_all_zero_quantiles_are_converted_to_IDSpline(metadata: pd.DataFrame,
                                                      INPUT_DIR,
                                                      tmp_path: Path):
    cn = cnp.CytoNorm()
    t = AsinhTransformer()
    fs = FlowSOM(n_clusters = 30) # way too many clusters, but we want that.
    cn.add_clusterer(fs)
    cn.add_transformer(t)
    coding_detectors = pd.read_csv(os.path.join(INPUT_DIR, "coding_detectors.txt"), header = None)[0].tolist()
    cn.run_fcs_data_setup(metadata = metadata,
                          input_directory = INPUT_DIR,
                          channels = coding_detectors,
                          output_directory = tmp_path)
    cn.run_clustering(cluster_cv_threshold = 2)
    cn.calculate_quantiles()
    # we make sure that we actually have all-zero quantiles
    mask = np.all(cn._expr_quantiles._expr_quantiles == 0, axis = (0))
    assert np.any(mask)
    # this should now run without error
    cn.calculate_splines()
    
    # we now check that all-zero quantiles have been converted
    # to identity splines
    for channel_idx, cluster_idx, batch_idx in np.argwhere(mask):
        channel = cn.channels[channel_idx]
        cluster = cn.clusters[cluster_idx]
        batch = cn.batches[batch_idx]
        spline = cn.splinefuncs.get_spline(batch, cluster, channel)
        
        assert spline.spline_calc_function.__qualname__ == "IdentitySpline"

    

