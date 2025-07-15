import pytest
from anndata import AnnData
import pandas as pd
import numpy as np

from cytonormpy._dataset._dataset import DataHandlerAnnData


def test_missing_colname(data_anndata: AnnData, DATAHANDLER_DEFAULT_KWARGS: dict):
    # dropping each required column in turn should KeyError
    for col in (
        DATAHANDLER_DEFAULT_KWARGS["reference_column"],
        DATAHANDLER_DEFAULT_KWARGS["batch_column"],
        DATAHANDLER_DEFAULT_KWARGS["sample_identifier_column"],
    ):
        ad = data_anndata.copy()
        ad.obs = ad.obs.drop(col, axis=1)
        with pytest.raises(KeyError):
            _ = DataHandlerAnnData(ad, **DATAHANDLER_DEFAULT_KWARGS)


def test_create_ref_data_df(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    df = dh._create_ref_data_df()
    assert isinstance(df, pd.DataFrame)
    # Reset index to expose the annotation columns
    cols = df.reset_index().columns
    rc = dh.metadata.reference_column
    bc = dh.metadata.batch_column
    sc = dh.metadata.sample_identifier_column
    assert {rc, bc, sc}.issubset(cols)
    # We expect 3 reference files × 1000 cells each = 3000 total rows
    assert df.shape[0] == 3000


def test_condense_metadata(data_anndata: AnnData, datahandleranndata: DataHandlerAnnData):
    obs = data_anndata.obs
    dh = datahandleranndata
    rc = dh.metadata.reference_column
    bc = dh.metadata.batch_column
    sc = dh.metadata.sample_identifier_column

    df = dh._condense_metadata(obs, rc, bc, sc)
    # sample‐identifier column must be unique
    assert not df[sc].duplicated().any()
    # dropping duplicates doesn't change shape
    assert df.shape == df.drop_duplicates().shape


def test_get_dataframe(datahandleranndata: DataHandlerAnnData, metadata: pd.DataFrame):
    dh = datahandleranndata
    fn = metadata[dh.metadata.sample_identifier_column].iloc[0]
    df = dh.get_dataframe(fn)
    # 1000 cells × 53 marker channels
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, len(dh.channels))
    # file_name, reference, batch should be index, not columns
    for col in (
        dh.metadata.sample_identifier_column,
        dh.metadata.reference_column,
        dh.metadata.batch_column,
    ):
        assert col not in df.columns


def test_find_and_get_array_indices(datahandleranndata: DataHandlerAnnData, metadata: pd.DataFrame):
    dh = datahandleranndata
    fn = metadata[dh.metadata.sample_identifier_column].iloc[0]

    obs_idxs = dh._find_obs_idxs(fn)
    assert isinstance(obs_idxs, pd.Index)
    arr_idxs = dh._get_array_indices(obs_idxs)
    assert isinstance(arr_idxs, np.ndarray)
    # round‐trip: indexing back should recover the same obs index
    recovered = dh.adata.obs.index[arr_idxs]
    pd.testing.assert_index_equal(recovered, obs_idxs)


def test_write_anndata(datahandleranndata: DataHandlerAnnData, metadata: pd.DataFrame):
    dh = datahandleranndata
    fn = metadata[dh.metadata.sample_identifier_column].iloc[0]

    # build a zero‐filled DataFrame matching the handler's channels
    zeros = np.zeros((1000, len(dh.channels)))
    df_zero = pd.DataFrame(zeros, columns=dh.channels)

    dh.write(fn, df_zero)

    # pull out the newly written layer for that file
    mask = dh.adata.obs[dh.metadata.sample_identifier_column] == fn
    subset = dh.adata[mask, :]
    layer_df = subset.to_df(layer=dh._key_added)

    # figure out which var‐indices were set
    idxs = dh._find_channel_indices_in_adata(df_zero.columns)
    changed = layer_df.iloc[:, idxs]
    # since we wrote zeros, the sum of each channel column must still be zero
    assert (changed.sum(axis=0) == 0).all()


def test_get_ref_data_df_and_subsampled(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata

    # get_ref_data_df should return the same as ref_data_df
    assert dh.get_ref_data_df().equals(dh.ref_data_df)

    # subsampled with default markers
    sub = dh.get_ref_data_df_subsampled(n=3000)
    assert isinstance(sub, pd.DataFrame)
    assert sub.shape[0] == 3000

    # too large n triggers ValueError
    with pytest.raises(ValueError):
        dh.get_ref_data_df_subsampled(n=10_000_000)


def test_marker_selection(
    datahandleranndata: DataHandlerAnnData,
    detectors: list[str],
    detector_subset: list[str],
    DATAHANDLER_DEFAULT_KWARGS: dict,
):
    dh = datahandleranndata

    # default ref_data_df has all marker columns
    full_n = dh.ref_data_df.shape[1]

    # selecting a subset
    sub = dh.get_ref_data_df(markers=detector_subset)
    assert sub.shape[1] == len(detector_subset)
    assert full_n != len(detector_subset)

    # subsampled + markers
    sub2 = dh.get_ref_data_df_subsampled(markers=detector_subset, n=10)
    assert sub2.shape == (10, len(detector_subset))


def test_find_marker_channels_and_technicals(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    all_det = dh._all_detectors
    markers = dh._find_marker_channels(all_det)
    tech = set(dh._flow_technicals + dh._cytof_technicals + dh._spectral_flow_technicals)
    # none of the returned markers should be in the combined technicals set
    assert not any(ch.lower() in tech for ch in markers)
