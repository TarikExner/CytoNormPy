import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from flowio import FlowData

from cytonormpy._dataset._dataset import DataHandlerFCS


def test_get_dataframe_fcs(datahandlerfcs: DataHandlerFCS, metadata: pd.DataFrame):
    fn = metadata["file_name"].iloc[0]
    df = datahandlerfcs.get_dataframe(fn)
    # Should be a 1000×53 DataFrame, indexed by (ref,batch,file_name)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 53)
    # columns should be channels only, not sample‐id
    assert "file_name" not in df.columns


def test_read_metadata_from_path_fcs(tmp_path, metadata: pd.DataFrame, INPUT_DIR: Path):
    # write CSV to disk, pass path into constructor
    fp = tmp_path / "meta.csv"
    metadata.to_csv(fp, index=False)
    dh = DataHandlerFCS(metadata=fp, input_directory=INPUT_DIR)
    # internal _metadata attr should equal the original table
    pd.testing.assert_frame_equal(metadata, dh.metadata.metadata)


def test_read_metadata_from_table_fcs(metadata: pd.DataFrame, INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata=metadata, input_directory=INPUT_DIR)
    pd.testing.assert_frame_equal(metadata, dh.metadata.metadata)


def test_metadata_missing_colname_fcs(metadata: pd.DataFrame, INPUT_DIR: Path):
    for col in ("reference", "file_name", "batch"):
        md = metadata.copy()
        bad = md.drop(col, axis=1)
        with pytest.raises(ValueError):
            _ = DataHandlerFCS(metadata=bad, input_directory=INPUT_DIR)


def test_write_fcs(
    tmp_path, datahandlerfcs: DataHandlerFCS, metadata: pd.DataFrame, INPUT_DIR: Path
):
    dh = datahandlerfcs
    fn = metadata["file_name"].iloc[0]
    # read raw events
    orig = FlowData(os.fspath(INPUT_DIR / fn))
    arr_orig = np.reshape(np.array(orig.events), (-1, orig.channel_count))

    # select only the channels the handler knows
    chdf = pd.DataFrame(arr_orig, columns=dh._all_detectors)[dh.channels]

    # perform write
    dh.write(file_name=fn, data=chdf, output_dir=tmp_path)

    out_fn = tmp_path / f"{dh._prefix}_{fn}"
    assert out_fn.exists()

    # re-read and compare
    new = FlowData(os.fspath(out_fn))
    arr_new = np.reshape(np.array(new.events), (-1, new.channel_count))

    # full event matrix should match original (unmodified channels get untouched)
    assert np.array_equal(arr_orig, arr_new)
    # metadata preserved
    assert set(orig.text.keys()).issubset(new.text.keys())
    assert set(orig.header.keys()).issubset(new.header.keys())
    # name, counts, channels match
    assert new.name == f"{dh._prefix}_{fn}"
    assert orig.channel_count == new.channel_count
    assert orig.event_count == new.event_count
    assert orig.analysis == new.analysis
    assert orig.channels == new.channels
