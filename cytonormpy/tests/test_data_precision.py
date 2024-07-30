import pytest
from anndata import AnnData
import pandas as pd
import numpy as np

from pathlib import Path
import os

import anndata as ad

import cytonormpy as cnp
from cytonormpy import AsinhTransformer, FCSFile

import flowio

# Module to test if R and python do the same thing.

def test_without_clustering_fcs(metadata: pd.DataFrame,
                                INPUT_DIR: Path,
                                tmpdir: Path):

    cn = cnp.CytoNorm()
    t = AsinhTransformer()
    cn.add_transformer(t)
    detectors = pd.read_csv(os.path.join(INPUT_DIR, "coding_detectors.txt"), header = None)[0].tolist()
    cn.run_fcs_data_setup(metadata = metadata,
                          input_directory = INPUT_DIR,
                          output_directory = tmpdir,
                          channels = detectors)

    cn.calculate_quantiles(n_quantiles = 99)
    cn.calculate_splines()
    cn.normalize_data()

    normalized_files = [
        "Norm_Gates_PTLG021_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG028_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG034_Unstim_Control_2.fcs",
    ]

    for file in normalized_files:
        r_version = FCSFile(INPUT_DIR, file)
        python_version = FCSFile(Path(tmpdir), file)

        assert r_version.channels.index.tolist() == python_version.channels.index.tolist()

        assert np.array_equal(
            r_version.original_events,
            python_version.original_events,
        )

def test_without_clustering_fcs_string_batch(metadata: pd.DataFrame,
                                             INPUT_DIR: Path,
                                             tmpdir: Path):
    metadata = metadata.copy()
    metadata["batch"] = [f"batch_{entry}" for entry in metadata["batch"].tolist()]
    cn = cnp.CytoNorm()
    t = AsinhTransformer()
    cn.add_transformer(t)
    detectors = pd.read_csv(os.path.join(INPUT_DIR, "coding_detectors.txt"), header = None)[0].tolist()
    cn.run_fcs_data_setup(metadata = metadata,
                          input_directory = INPUT_DIR,
                          output_directory = tmpdir,
                          channels = detectors)

    cn.calculate_quantiles(n_quantiles = 99)
    cn.calculate_splines()
    cn.normalize_data()

    normalized_files = [
        "Norm_Gates_PTLG021_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG028_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG034_Unstim_Control_2.fcs",
    ]

    for file in normalized_files:
        r_version = FCSFile(INPUT_DIR, file)
        python_version = FCSFile(Path(tmpdir), file)

        assert r_version.channels.index.tolist() == python_version.channels.index.tolist()

        assert np.array_equal(
            r_version.original_events,
            python_version.original_events,
        )


def _create_anndata(input_dir, file_list):
    adatas = []
    for file in file_list:
        fcs_data = flowio.FlowData(os.path.join(input_dir, file))
        events = np.reshape(
            np.array(fcs_data.events, dtype = np.float64),
            (-1, fcs_data.channel_count),
        )
        fcs = FCSFile(input_directory = input_dir,
                      file_name = file)

        md_row = np.array([file.strip("Norm_")])
        obs = np.repeat(
            md_row,
            events.shape[0],
            axis = 0
        )
        var_frame = fcs.channels
        obs_frame = pd.DataFrame(
            data = obs,
            columns = ["file_name"],
            index = pd.Index([str(i) for i in range(events.shape[0])])
        )
        adata = ad.AnnData(
            obs = obs_frame,
            var = var_frame,
            layers = {"normalized": events}
        )
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adatas.append(adata)

    dataset = ad.concat(adatas, axis = 0, join = "outer", merge = "same")
    dataset.obs = dataset.obs.astype(str)
    dataset.var = dataset.var.astype(str)
    dataset.var_names_make_unique()
    dataset.obs_names_make_unique()

    return dataset
    
def test_without_clustering_anndata(data_anndata: AnnData,
                                    INPUT_DIR: Path):
    r_normalized_files = [
        "Norm_Gates_PTLG021_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG028_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG034_Unstim_Control_2.fcs",
    ]

    r_anndata = _create_anndata(INPUT_DIR, r_normalized_files)

    data_anndata.obs["batch"] = data_anndata.obs["batch"].astype(np.int8)
    data_anndata.obs["batch"] = data_anndata.obs["batch"].astype("category")


    cn = cnp.CytoNorm()
    t = AsinhTransformer()
    cn.add_transformer(t)
    detectors = pd.read_csv(os.path.join(INPUT_DIR, "coding_detectors.txt"), header = None)[0].tolist()
    cn.run_anndata_setup(adata = data_anndata,
                         layer = "compensated",
                         channels = detectors,
                         key_added = "normalized")
    cn.calculate_quantiles(n_quantiles = 99)
    cn.calculate_splines()
    cn.normalize_data()

    assert "normalized" in data_anndata.layers.keys()

    comp_data = data_anndata[data_anndata.obs["reference"] == "other",:].copy()

    assert comp_data.obs["file_name"].unique().tolist() == r_anndata.obs["file_name"].unique().tolist()
    assert comp_data.obs["file_name"].tolist() == r_anndata.obs["file_name"].tolist()
    assert comp_data.shape == r_anndata.shape

    np.testing.assert_array_almost_equal(
        np.array(r_anndata.layers["normalized"]),
        np.array(comp_data.layers["normalized"]),
        decimal = 3
    )

def test_without_clustering_anndata_string_batch(data_anndata: AnnData,
                                                 INPUT_DIR: Path):
    r_normalized_files = [
        "Norm_Gates_PTLG021_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG028_Unstim_Control_2.fcs",
        "Norm_Gates_PTLG034_Unstim_Control_2.fcs",
    ]

    r_anndata = _create_anndata(INPUT_DIR, r_normalized_files)

    data_anndata.obs["batch"] = [f"batch_{entry}" for entry in data_anndata.obs["batch"].tolist()]
    data_anndata.obs["batch"] = data_anndata.obs["batch"].astype("category")


    cn = cnp.CytoNorm()
    t = AsinhTransformer()
    cn.add_transformer(t)
    detectors = pd.read_csv(os.path.join(INPUT_DIR, "coding_detectors.txt"), header = None)[0].tolist()
    cn.run_anndata_setup(adata = data_anndata,
                         layer = "compensated",
                         channels = detectors,
                         key_added = "normalized")
    cn.calculate_quantiles(n_quantiles = 99)
    cn.calculate_splines()
    cn.normalize_data()

    assert "normalized" in data_anndata.layers.keys()

    comp_data = data_anndata[data_anndata.obs["reference"] == "other",:].copy()

    assert comp_data.obs["file_name"].unique().tolist() == r_anndata.obs["file_name"].unique().tolist()
    assert comp_data.obs["file_name"].tolist() == r_anndata.obs["file_name"].tolist()
    assert comp_data.shape == r_anndata.shape

    np.testing.assert_array_almost_equal(
        np.array(r_anndata.layers["normalized"]),
        np.array(comp_data.layers["normalized"]),
        decimal = 3
    )
