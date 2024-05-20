import pytest
import pandas as pd
import os
import numpy as np
from pathlib import Path
from flowio import FlowData
from cytonormpy._dataset._dataset import DataHandlerFCS
from cytonormpy._dataset._fcs_file import FCSFile


def test_attribute_assignments(metadata: pd.DataFrame,
                               INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata = metadata,
                        input_directory = INPUT_DIR)
    assert metadata.equals(dh._metadata)
    assert dh._input_dir == INPUT_DIR
    assert dh._reference_column == "reference"
    assert dh._batch_column == "batch"
    assert dh._sample_identifier_column == "file_name"
    assert dh._reference_value == "ref"
    assert isinstance(dh.ref_data_df, pd.DataFrame)
    assert dh._truncate_max_range is True
    assert isinstance(dh.channels, list)
    assert dh._output_dir == INPUT_DIR


def test_create_ref_data_df(datahandlerfcs: DataHandlerFCS):
    dh = datahandlerfcs
    df = dh._create_ref_data_df()
    assert isinstance(df, pd.DataFrame)
    df = df.reset_index()
    assert all(
        k in df.columns
        for k in [dh._reference_column, dh._batch_column,
                  dh._sample_identifier_column]
    )
    assert df.shape[0] == 3000


def test_get_dataframe(datahandlerfcs: DataHandlerFCS,
                       metadata: pd.DataFrame):
    req_file = metadata["file_name"].tolist()[0]
    dh = datahandlerfcs
    df = dh.get_dataframe(req_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 53)
    assert "file_name" not in df.columns

    df = dh.get_dataframe(req_file, raw = True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 55)
    assert "file_name" not in df.columns

    df = dh.get_dataframe(req_file, raw = True, annot_file_name = True)
    assert dh._sample_identifier_column == "file_name"
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 55 + 1)
    assert "file_name" in df.columns

    df = dh.get_dataframe(req_file, raw = False, annot_file_name = True)
    assert dh._sample_identifier_column == "file_name"
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 53 + 1)
    assert "file_name" in df.columns


def test_fcs_to_df(datahandlerfcs: DataHandlerFCS,
                   metadata: pd.DataFrame):
    dh = datahandlerfcs
    req_file = metadata["file_name"].tolist()[0]
    df = dh._fcs_to_df(file_name = req_file)
    assert isinstance(df, pd.DataFrame)
    assert "reference" in df.index.names
    assert "batch" in df.index.names
    assert "file_name" in df.index.names
    assert df.shape == (1000, 55)  # original data, no subset
    df = df.reset_index()
    assert all(
        k in df.columns
        for k in [dh._reference_column,
                  dh._batch_column,
                  dh._sample_identifier_column]
    )
    assert df.shape[0] == 1000


def test_read_fcs_file(datahandlerfcs: DataHandlerFCS,
                       metadata: pd.DataFrame):
    dh = datahandlerfcs
    req_file = metadata["file_name"].tolist()[0]
    fcs = dh._read_fcs_file(req_file)
    df = fcs.to_df()
    assert isinstance(fcs, FCSFile)
    assert isinstance(fcs.channels, pd.DataFrame)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 55)


def test_read_metadata_from_path(tmp_path,
                                 metadata: pd.DataFrame,
                                 INPUT_DIR: Path):
    file_path = Path(os.path.join(tmp_path, "metadata.csv"))
    metadata.to_csv(file_path, index = False)
    dataset = DataHandlerFCS(metadata = file_path,
                             input_directory = INPUT_DIR)
    assert metadata.equals(dataset._metadata)


def test_read_metadata_from_table(metadata: pd.DataFrame,
                                  INPUT_DIR: Path):
    dataset = DataHandlerFCS(metadata = metadata,
                             input_directory = INPUT_DIR)
    assert metadata.equals(dataset._metadata)


def test_metadata_missing_colname(metadata: pd.DataFrame,
                                  INPUT_DIR: Path):
    md = metadata.drop("reference", axis = 1)
    with pytest.raises(ValueError):
        _ = DataHandlerFCS(metadata = md,
                           input_directory = INPUT_DIR)
    md = metadata.drop("file_name", axis = 1)
    with pytest.raises(ValueError):
        _ = DataHandlerFCS(metadata = md,
                           input_directory = INPUT_DIR)

    md = metadata.drop("batch", axis = 1)
    with pytest.raises(ValueError):
        _ = DataHandlerFCS(metadata = md,
                           input_directory = INPUT_DIR)


def test_write_fcs(tmp_path,
                   datahandlerfcs: DataHandlerFCS,
                   metadata: pd.DataFrame,
                   INPUT_DIR: Path):
    dh = datahandlerfcs
    req_file = metadata["file_name"].tolist()[0]
    fcs = FlowData(os.path.join(INPUT_DIR, req_file))
    original_data = np.reshape(np.array(fcs.events),
                               (-1, fcs.channel_count))
    ch_spec_data = original_data[:, dh._channel_indices]

    dh.write(req_file,
             output_dir = tmp_path,
             data = ch_spec_data)

    assert os.path.isfile(os.path.join(tmp_path,
                                       f"{dh._prefix}_{req_file}"))

    reread = FlowData(
        os.path.join(tmp_path,
                     f"{dh._prefix}_{req_file}")
    )

    assert np.array_equal(
        original_data,
        np.reshape(np.array(reread.events),
                   (-1, reread.channel_count))
    )
    assert all(k in list(reread.text.keys())
               for k in list(fcs.text.keys()))
    assert all(k in list(reread.header.keys())
               for k in list(fcs.header.keys()))
    assert reread.name == f"{dh._prefix}_{req_file}"
    assert fcs.channel_count == reread.channel_count
    assert fcs.event_count == reread.event_count
    assert fcs.analysis == reread.analysis
    assert fcs.channels == reread.channels
