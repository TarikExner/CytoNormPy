import pytest
import pandas as pd
import os
import numpy as np
from pathlib import Path
from flowio import FlowData
from cytonormpy._dataset._dataset import DataHandlerFCS


def test_get_dataframe(datahandlerfcs: DataHandlerFCS,
                       metadata: pd.DataFrame):
    req_file = metadata["file_name"].tolist()[0]
    dh = datahandlerfcs
    df = dh.get_dataframe(req_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 53)
    assert "file_name" not in df.columns

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
    ch_spec_data = pd.DataFrame(data = original_data,
                                columns = dh._all_detectors,
                                index = list(range(original_data.shape[0])))
    ch_spec_data = pd.DataFrame(ch_spec_data[dh.channels])

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
