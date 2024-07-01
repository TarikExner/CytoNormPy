import os
from pathlib import Path
import pandas as pd
from anndata import AnnData
import anndata as ad
import numpy as np

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

    fcs_files = [file for file in os.listdir(fcs_dir)
                 if file.endswith(".fcs")]
    adatas = []
    metadata = pd.read_csv(os.path.join(fcs_dir, "metadata_sid.csv"))
    for file in fcs_files:
        fcs = FCSFile(input_directory = fcs_dir,
                      file_name = file)
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


def example_cytonorm():
    data_dir = Path(__file__).parent.parent
    metadata = pd.read_csv(os.path.join(data_dir, "_resources/metadata_sid.csv"))
    cn = CytoNorm()
    fs = FlowSOM(n_clusters = 10)
    t = AsinhTransformer(cofactors = 5)
    cn.add_clusterer(fs)
    cn.add_transformer(t)
    cn.run_fcs_data_setup(
        input_directory = os.path.join(data_dir, "_resources"),
        metadata = metadata
    )
    cn.run_clustering(cluster_cv_threshold = 2)
    cn.calculate_quantiles()
    cn.calculate_splines(goal = "batch_mean")
    cn.normalize_data()

    return cn
    
    
