import pytest
import pandas as pd

import cytonormpy as cnp
import numpy as np


CELL_LABELS = ["T_cells", "B_cells", "NK_cells", "Monocytes", "Neutrophils"]

def _generate_cell_labels(n: int = 1000):
    return np.random.choice(CELL_LABELS, n, replace = True)

def test_data_setup_fcs(INPUT_DIR,
                        metadata: pd.DataFrame,
                        tmpdir):

    cn = cnp.CytoNorm()
    t = cnp.AsinhTransformer()
    cn.add_transformer(t)
    cn.run_fcs_data_setup(input_directory = INPUT_DIR,
                          metadata = metadata,
                          channels = "markers",
                          output_directory = tmpdir)
    cn.calculate_quantiles()
    cn.calculate_splines()
    cn.normalize_data()

    cn.calculate_mad(groupby = "file_name")
    df = cn.mad_frame
    assert all(ch in df.columns for ch in cn._datahandler.channels)
    assert all(entry in df.index.names for entry in ["file_name", "origin", "label"])
    assert df.shape[0] == len(cn._datahandler.validation_file_names)*2

    cn.calculate_mad(groupby = "label")
    df = cn.mad_frame
    assert all(ch in df.columns for ch in cn._datahandler.channels)
    assert all(entry in df.index.names for entry in ["origin", "label"])
    assert df.index.get_level_values("label").nunique() == 1
    assert df.index.get_level_values("origin").nunique() == 2
    assert df.shape[0] == 2

    label_dict = {}
    for file in cn._datahandler.validation_file_names:
        labels = _generate_cell_labels()
        label_dict[file] = labels
        label_dict["Norm_" + file] = labels

    cn.calculate_mad(groupby = ["file_name", "label"], cell_labels = label_dict)
    df = cn.mad_frame
    assert all(ch in df.columns for ch in cn._datahandler.channels)
    assert all(entry in df.index.names for entry in ["file_name", "origin", "label"])
    assert all(
        label in df.index.get_level_values("label").unique().tolist()
        for label in CELL_LABELS + ["all_cells"]
    )
    assert df.shape[0] == len(cn._datahandler.validation_file_names)*2*(len(CELL_LABELS)+1)


def test_data_setup_anndata(data_anndata):

    data_anndata.obs["cell_type"] = _generate_cell_labels(data_anndata.shape[0])
    data_anndata.obs["batch"] = data_anndata.obs["batch"].astype(np.int8)

    cn = cnp.CytoNorm()
    t = cnp.AsinhTransformer()
    cn.add_transformer(t)
    cn.run_anndata_setup(adata = data_anndata)
    cn.calculate_quantiles()
    cn.calculate_splines()
    cn.normalize_data()

    cn.calculate_mad(groupby = "file_name")
    df = cn.mad_frame
    assert all(ch in df.columns for ch in cn._datahandler.channels)
    assert all(entry in df.index.names for entry in ["file_name", "origin", "label"])
    assert df.shape[0] == len(cn._datahandler.validation_file_names)*2

    cn.calculate_mad(groupby = "label")
    df = cn.mad_frame
    assert all(ch in df.columns for ch in cn._datahandler.channels)
    assert all(entry in df.index.names for entry in ["origin", "label"])
    assert df.shape[0] == 2

    cn.calculate_mad(groupby = ["file_name", "label"], cell_labels = "cell_type")
    df = cn.mad_frame
    assert all(ch in df.columns for ch in cn._datahandler.channels)
    assert all(entry in df.index.names for entry in ["file_name", "origin", "label"])
    assert all(
        label in df.index.get_level_values("label").unique().tolist()
        for label in CELL_LABELS + ["all_cells"]
    )
    assert df.shape[0] == len(cn._datahandler.validation_file_names)*2*(len(CELL_LABELS)+1)


def test_r_python_mad():
    from scipy.stats import median_abs_deviation

    arr = np.arange(10)

    r_val = 3.7065
    assert round(median_abs_deviation(arr, scale = "normal"), 4) == r_val


    






