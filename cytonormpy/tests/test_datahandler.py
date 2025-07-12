import pytest
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import numpy as np
from anndata import AnnData
from cytonormpy._dataset._dataset import DataHandlerFCS, DataHandlerAnnData


def test_technical_setters_and_append(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    dh.flow_technicals = ["foo"]
    assert dh.flow_technicals == ["foo"]
    dh.append_flow_technicals("bar")
    assert "bar" in dh.flow_technicals
    dh.cytof_technicals = ["x"]
    assert dh.cytof_technicals == ["x"]
    dh.append_cytof_technicals("y")
    assert "y" in dh.cytof_technicals
    dh.spectral_flow_technicals = ["p"]
    assert dh.spectral_flow_technicals == ["p"]
    dh.append_spectral_flow_technicals("q")
    assert "q" in dh.spectral_flow_technicals


def test_correct_df_shape_all_channels(metadata: pd.DataFrame, INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata=metadata, input_directory=INPUT_DIR, channels="all")
    assert dh.ref_data_df.shape == (3000, 55)


def test_correct_df_shape_all_channels_anndata(
    data_anndata: AnnData, DATAHANDLER_DEFAULT_KWARGS: dict
):
    kwargs = DATAHANDLER_DEFAULT_KWARGS.copy()
    kwargs["channels"] = "all"
    dh = DataHandlerAnnData(data_anndata, **kwargs)
    assert dh.ref_data_df.shape == (3000, 55)


def test_correct_df_shape_markers(datahandlerfcs: DataHandlerFCS):
    # Time and Event_length are excluded
    assert datahandlerfcs.ref_data_df.shape == (3000, 53)


def test_correct_df_shape_markers_anndata(datahandleranndata: DataHandlerAnnData):
    # Time and Event_length are excluded
    assert datahandleranndata.ref_data_df.shape == (3000, 53)


def test_correct_df_shape_channellist(
    metadata: pd.DataFrame, detectors: list[str], INPUT_DIR: Path
):
    dh = DataHandlerFCS(metadata=metadata, input_directory=INPUT_DIR, channels=detectors[:30])
    assert dh.ref_data_df.shape == (3000, 30)


def test_correct_df_shape_channellist_anndata(
    data_anndata: AnnData, detectors: list[str], DATAHANDLER_DEFAULT_KWARGS: dict
):
    kwargs = DATAHANDLER_DEFAULT_KWARGS.copy()
    kwargs["channels"] = detectors[:30]
    dh = DataHandlerAnnData(data_anndata, **kwargs)
    assert dh.ref_data_df.shape == (3000, 30)


def test_correct_channel_indices_markers_fcs(metadata: pd.DataFrame, INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata=metadata, input_directory=INPUT_DIR, channels="markers")
    # get raw fcs channels from the first file
    raw = dh._provider._reader.parse_fcs_df(metadata["file_name"].iloc[0])
    fcs_channels = raw.columns.tolist()
    idxs = dh._channel_indices
    selected = [fcs_channels[i] for i in idxs]
    assert dh.ref_data_df.columns.tolist() == selected


def test_correct_channel_indices_markers_anndata(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    adata_ch = dh.adata.var_names.tolist()
    idxs = dh._channel_indices
    selected = [adata_ch[i] for i in idxs]
    assert dh.ref_data_df.columns.tolist() == selected


def test_correct_channel_indices_list_fcs(
    metadata: pd.DataFrame, detectors: list[str], INPUT_DIR: Path
):
    subset = detectors[:30]
    dh = DataHandlerFCS(
        metadata=metadata,
        input_directory=INPUT_DIR,
        channels=subset,
    )
    raw = dh._provider._reader.parse_fcs_df(metadata["file_name"].iloc[0])
    fcs_channels = raw.columns.tolist()
    idxs = dh._channel_indices
    selected = [fcs_channels[i] for i in idxs]
    assert dh.ref_data_df.columns.tolist() == selected


def test_correct_channel_indices_list_anndata(
    data_anndata: AnnData, detectors: list[str], DATAHANDLER_DEFAULT_KWARGS: dict
):
    subset = detectors[:30]
    kwargs = DATAHANDLER_DEFAULT_KWARGS.copy()
    kwargs["channels"] = subset
    dh = DataHandlerAnnData(data_anndata, **kwargs)
    ch = dh.adata.var_names.tolist()
    idxs = dh._channel_indices
    selected = [ch[i] for i in idxs]
    assert dh.ref_data_df.columns.tolist() == selected


def test_ref_data_df_index_multiindex(datahandlerfcs: DataHandlerFCS):
    df = datahandlerfcs.ref_data_df
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["reference", "batch", "file_name"]


def test_ref_data_df_index_multiindex_anndata(datahandleranndata: DataHandlerAnnData):
    df = datahandleranndata.ref_data_df
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["reference", "batch", "file_name"]


def test_get_batch_anndata(datahandleranndata: DataHandlerAnnData, metadata: pd.DataFrame):
    dh = datahandleranndata
    fn = metadata["file_name"].iloc[0]
    expected = metadata.loc[metadata.file_name == fn, "batch"].iloc[0]
    got = dh.metadata.get_batch(fn)
    assert str(got) == str(expected)


def test_find_corresponding_reference_file_anndata(
    datahandleranndata: DataHandlerAnnData, metadata: pd.DataFrame
):
    dh = datahandleranndata
    fn = metadata["file_name"].iloc[1]
    batch = dh.metadata.get_batch(fn)
    others = metadata.loc[metadata.batch == int(batch), "file_name"].tolist()
    corr = [x for x in others if x != fn][0]
    assert dh.metadata.get_corresponding_reference_file(fn) == corr


def test_get_corresponding_ref_dataframe(
    datahandleranndata: DataHandlerAnnData, metadata: pd.DataFrame
):
    dh = datahandleranndata
    fn = metadata["file_name"].iloc[1]
    ref_df = dh.get_corresponding_ref_dataframe(fn)
    sample_df = dh.get_dataframe(fn)
    # reference file has same shape but different content
    assert ref_df.shape == sample_df.shape
    # first 14 rows differ
    assert not np.allclose(ref_df.iloc[:14].values, sample_df.iloc[:14].values)


def test_get_ref_data_df_alias(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    assert dh.ref_data_df.equals(dh.get_ref_data_df())


def test_get_ref_data_df_subsampled_length(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    sub = dh.get_ref_data_df_subsampled(n=300)
    assert sub.shape[0] == 300


def test_get_ref_data_df_subsampled_too_large(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    with pytest.raises(ValueError):
        dh.get_ref_data_df_subsampled(n=10_000_000)


def test_subsample_df_method(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    df = dh.ref_data_df
    sub = dh._subsample_df(df, n=300)
    assert sub.shape[0] == 300


def test_artificial_ref_on_relabeled_batch_anndata(
    data_anndata: AnnData, DATAHANDLER_DEFAULT_KWARGS: dict
):
    # relabel so chosen batch has no true reference samples
    ad = data_anndata.copy()
    dh_kwargs = DATAHANDLER_DEFAULT_KWARGS.copy()
    dh_kwargs["n_cells_reference"] = 500

    # extract metadata column names
    rc = dh_kwargs["reference_column"]
    rv = dh_kwargs["reference_value"]
    bc = dh_kwargs["batch_column"]
    sc = dh_kwargs["sample_identifier_column"]

    # pick a batch and relabel its ref entries
    target = ad.obs[bc].unique()[0]
    mask = (ad.obs[bc] == target) & (ad.obs[rc] == rv)
    ad.obs.loc[mask, rc] = "other"

    dh = DataHandlerAnnData(ad, **dh_kwargs)
    df = dh.ref_data_df

    # EXPECT: this batch appears in reference_assembly_dict
    expected_files = ad.obs.loc[ad.obs[bc] == target, sc].unique().tolist()
    assert int(target) in dh.metadata.reference_assembly_dict
    assert set(dh.metadata.reference_assembly_dict[int(target)]) == set(expected_files)

    # EXPECT: exactly n_cells_reference rows for that batch
    idx_batch = df.index.get_level_values(dh.metadata.batch_column)
    n_observed = (idx_batch == int(target)).sum()
    assert n_observed == 500, idx_batch

    # EXPECT: sample‐identifier level all set to artificial label
    idx_samp = df.index.get_level_values(dh.metadata.sample_identifier_column)
    artificial = f"__B_{target}_CYTONORM_GENERATED__"
    unique_vals = set(idx_samp.unique())
    assert artificial in unique_vals
    assert idx_samp.tolist().count(artificial) == 500


def test_artificial_ref_on_relabeled_batch_fcs(metadata: pd.DataFrame, INPUT_DIR: str):
    # relabel so chosen batch has no true reference samples
    md = metadata.copy()
    rc, rv, bc, sc = "reference", "ref", "batch", "file_name"
    target = md[bc].unique()[0]
    md.loc[(md[bc] == target) & (md[rc] == rv), rc] = "other"

    # build handler with n_cells_reference
    N = 500
    dh = DataHandlerFCS(
        metadata=md,
        input_directory=INPUT_DIR,
        channels="markers",
        n_cells_reference=N,
        reference_column=rc,
        reference_value=rv,
        batch_column=bc,
        sample_identifier_column=sc,
    )
    df = dh.ref_data_df

    # EXPECT: batch in reference_assembly_dict with all its files
    expected_files = md.loc[md[bc] == target, sc].tolist()
    assert target in dh.metadata.reference_assembly_dict
    assert set(dh.metadata.reference_assembly_dict[target]) == set(expected_files)

    # EXPECT: exactly n_cells_reference rows for that batch
    idx_batch = df.index.get_level_values(dh.metadata.batch_column)
    n_observed = (idx_batch == target).sum()
    assert n_observed == 500

    # EXPECT: sample‐identifier level all set to artificial label
    idx_samp = df.index.get_level_values(dh.metadata.sample_identifier_column)
    artificial = f"__B_{target}_CYTONORM_GENERATED__"
    unique_vals = set(idx_samp.unique())
    assert artificial in unique_vals
    assert idx_samp.tolist().count(artificial) == 500


def test_find_marker_channels_excludes_technicals(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    all_det = dh._all_detectors
    markers = dh._find_marker_channels(all_det)
    tech = set(dh._flow_technicals + dh._cytof_technicals + dh._spectral_flow_technicals)
    assert not any(ch.lower() in tech for ch in markers)


def test_add_file_fcs_updates_metadata_and_provider(
    metadata: pd.DataFrame, INPUT_DIR: Path, DATAHANDLER_DEFAULT_KWARGS: dict
):
    dh = DataHandlerFCS(
        metadata=metadata.copy(),
        input_directory=INPUT_DIR,
        channels="markers",
    )
    new_file = "newfile.fcs"
    dh.add_file(new_file, batch=1)
    assert new_file in dh.metadata.metadata.file_name.values
    # provider.metadata should point to same Metadata instance
    assert dh._provider.metadata is dh.metadata


def test_add_file_anndata_updates_metadata_and_layer(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    new_file = "newfile.fcs"
    dh.add_file(new_file, batch=1)
    # metadata and provider metadata updated
    assert new_file in dh.metadata.metadata.file_name.values
    assert dh._provider.metadata is dh.metadata


def test_string_batch_conversion_fcs(
    metadata: pd.DataFrame, INPUT_DIR: Path, DATAHANDLER_DEFAULT_KWARGS: dict
):
    md = metadata.copy()
    md["batch"] = [f"batch_{b}" for b in md.batch]
    dh = DataHandlerFCS(
        metadata=md,
        input_directory=INPUT_DIR,
        channels="markers",
    )
    new_md = dh.metadata
    assert "original_batch" in new_md.metadata.columns
    assert is_numeric_dtype(new_md.metadata.batch)


def test_string_batch_conversion_anndata(data_anndata: AnnData, DATAHANDLER_DEFAULT_KWARGS: dict):
    ad = data_anndata.copy()
    ad.obs["batch"] = [f"batch_{b}" for b in ad.obs.batch]
    kwargs = DATAHANDLER_DEFAULT_KWARGS.copy()
    dh = DataHandlerAnnData(**kwargs, adata=ad)
    new_md = dh.metadata
    assert "original_batch" in new_md.metadata.columns
    assert is_numeric_dtype(new_md.metadata.batch)


def test_marker_selection_filters_columns(
    datahandleranndata: DataHandlerAnnData,
    detectors: list[str],
    detector_subset: list[str],
    DATAHANDLER_DEFAULT_KWARGS: dict,
):
    dh = datahandleranndata
    # get only subset
    df = dh.get_ref_data_df(markers=detector_subset)
    assert df.shape[1] == len(detector_subset)
    assert dh.ref_data_df.shape[1] != len(detector_subset)


def test_marker_selection_subsampled_filters_and_counts(
    datahandleranndata: DataHandlerAnnData,
    detectors: list[str],
    detector_subset: list[str],
    DATAHANDLER_DEFAULT_KWARGS: dict,
):
    dh = datahandleranndata
    df = dh.get_ref_data_df_subsampled(markers=detector_subset, n=10)
    assert df.shape == (10, len(detector_subset))


def test_no_reference_files_all_artificial_fcs(metadata: pd.DataFrame, INPUT_DIR: Path):
    # Relabel every sample as non‐reference
    md = metadata.copy()
    md["reference"] = "other"  # nothing equals the default 'ref'
    n_cells_reference = 200

    dh = DataHandlerFCS(
        metadata=md,
        input_directory=INPUT_DIR,
        channels="markers",
        n_cells_reference=n_cells_reference,
    )

    df = dh.ref_data_df
    # Expect one artificial block per batch
    unique_batches = md["batch"].unique()
    assert df.shape[0] == n_cells_reference * len(unique_batches)

    # And each artificial block should carry exactly n_cells_reference rows
    samp_col = dh.metadata.sample_identifier_column
    idx_samples = df.index.get_level_values(samp_col)
    for batch in unique_batches:
        label = f"__B_{batch}_CYTONORM_GENERATED__"
        assert (idx_samples == label).sum() == n_cells_reference


def test_no_reference_files_all_artificial_anndata(
    data_anndata: AnnData, DATAHANDLER_DEFAULT_KWARGS: dict
):
    # Copy the AnnData and relabel all obs as non‐reference
    ad = data_anndata.copy()
    kw = DATAHANDLER_DEFAULT_KWARGS.copy()
    rc = kw["reference_column"]
    ad.obs[rc] = "other"  # override every row

    n_cells_reference = 150
    kw["n_cells_reference"] = n_cells_reference

    dh = DataHandlerAnnData(adata=ad, **kw)

    df = dh.ref_data_df
    # One artificial block per batch
    unique_batches = ad.obs[kw["batch_column"]].unique()
    assert df.shape[0] == n_cells_reference * len(unique_batches)

    # Each block labeled correctly at the sample‐identifier level
    samp_col = kw["sample_identifier_column"]
    idx_samples = df.index.get_level_values(samp_col)
    for batch in unique_batches:
        label = f"__B_{batch}_CYTONORM_GENERATED__"
        assert (idx_samples == label).sum() == n_cells_reference
