import os
from pathlib import Path
import pandas as pd
from anndata import AnnData
import anndata as ad
import numpy as np

import tempfile
import shutil

from ._cytonorm import CytoNorm
from .._clustering import FlowSOM
from .._dataset import FCSFile
from .._transformation import AsinhTransformer

def example_anndata() -> AnnData:
    HERE = Path(__file__).parent
    pkg_folder = HERE.parent
    fcs_dir = os.path.join(pkg_folder, "_resources")
    adata_file = Path(os.path.join(fcs_dir, "test_adata.h5ad"))

    if os.path.isfile(adata_file):
        return ad.read_h5ad(adata_file)

    adatas = []
    metadata = pd.read_csv(os.path.join(fcs_dir, "metadata_sid.csv"))
    for file in metadata["file_name"].tolist():
        fcs = FCSFile(input_directory = fcs_dir,
                      file_name = file,
                      truncate_max_range = True)
        events = fcs.original_events
        md_row = metadata.loc[
            metadata["file_name"] == file, :
        ].to_numpy()
        obs = np.repeat(
            md_row,
            events.shape[0],
            axis = 0
        )
        var_frame = fcs.channels
        obs_frame = pd.DataFrame(
            data = obs,
            columns = metadata.columns,
            index = pd.Index([str(i) for i in range(events.shape[0])])
        )
        adata = ad.AnnData(
            obs = obs_frame,
            var = var_frame,
            layers = {"compensated": events}
        )
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        adatas.append(adata)

    dataset = ad.concat(adatas, axis = 0, join = "outer", merge = "same")
    dataset.obs = dataset.obs.astype(str)
    dataset.var = dataset.var.astype(str)
    dataset.obs_names_make_unique()
    dataset.var_names_make_unique()
    dataset.write(adata_file)
    return dataset

def _generate_cell_labels(n: int):
    all_cell_labels = ["T_cells", "B_cells", "NK_cells", "Monocytes", "Neutrophils"]
    np.random.seed(187)
    return np.random.choice(all_cell_labels, n, replace = True)

def example_cytonorm(use_clustering: bool = False):
    tmp_dir = tempfile.mkdtemp()
    data_dir = Path(__file__).parent.parent
    metadata = pd.read_csv(os.path.join(data_dir, "_resources/metadata_sid.csv"))
    channels = pd.read_csv(os.path.join(data_dir, "_resources/coding_detectors.txt"), header = None)[0].tolist()
    original_files = metadata.loc[metadata["reference"] == "other", "file_name"].to_list()
    normalized_files = ["Norm_" + file_name for file_name in original_files]
    cell_labels = {
        file: _generate_cell_labels(1000)
        for file in original_files + normalized_files
    }
    cn = CytoNorm()
    if use_clustering:
        fs = FlowSOM(n_clusters = 10)
        cn.add_clusterer(fs)
    t = AsinhTransformer(cofactors = 5)
    cn.add_transformer(t)
    cn.run_fcs_data_setup(
        input_directory = os.path.join(data_dir, "_resources"),
        metadata = metadata,
        output_directory = tmp_dir,
        channels = channels
    )
    if use_clustering:
        cn.run_clustering(cluster_cv_threshold = 2)
    cn.calculate_quantiles()
    cn.calculate_splines(goal = "batch_mean")
    cn.normalize_data()
    cn.calculate_mad(groupby = ["file_name", "label"], cell_labels = cell_labels)
    cn.calculate_emd(cell_labels = cell_labels)

    shutil.rmtree(tmp_dir)

    return cn
    
    
