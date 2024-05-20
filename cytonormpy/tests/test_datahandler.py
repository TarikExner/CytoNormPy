import pytest
import pandas as pd
from pathlib import Path
import numpy as np
from anndata import AnnData
from cytonormpy._dataset._dataset import DataHandlerFCS, DataHandlerAnnData


def test_conclusive_reference_values(metadata: pd.DataFrame,
                                     INPUT_DIR: Path):
    md = metadata
    md.loc[
        md["file_name"] == "Gates_PTLG021_Unstim_Control_1.fcs",
        "reference"
    ] = "what"
    with pytest.raises(ValueError):
        _ = DataHandlerFCS(metadata = md,
                           input_directory = INPUT_DIR)


def test_conclusive_reference_values_anndata(data_anndata: AnnData,
                                             DATAHANDLER_DEFAULT_KWARGS: dict):
    adata = data_anndata
    adata.obs["reference"] = adata.obs["reference"].astype(str)
    adata.obs.loc[
        adata.obs["batch"] == "3",
        "reference"
    ] = "additional_ref_value"
    with pytest.raises(ValueError):
        _ = DataHandlerAnnData(adata, **DATAHANDLER_DEFAULT_KWARGS)


def test_all_batches_have_reference(metadata: pd.DataFrame,
                                    INPUT_DIR: Path):
    md = metadata
    md.loc[
        md["file_name"] == "Gates_PTLG021_Unstim_Control_1.fcs",
        "reference"
    ] = "other"
    with pytest.raises(ValueError):
        _ = DataHandlerFCS(metadata = md,
                           input_directory = INPUT_DIR)


def test_all_batches_have_reference_anndata(data_anndata: AnnData,
                                            DATAHANDLER_DEFAULT_KWARGS):
    adata = data_anndata
    x = DataHandlerAnnData(adata, **DATAHANDLER_DEFAULT_KWARGS)
    assert isinstance(x, DataHandlerAnnData)


def test_all_batches_have_reference_false(data_anndata: AnnData,
                                          DATAHANDLER_DEFAULT_KWARGS: dict):
    adata = data_anndata
    adata.obs["reference"] = adata.obs["reference"].astype(str)
    adata.obs.loc[
        adata.obs["batch"] == "3",
        "reference"
    ] = "other"
    with pytest.raises(ValueError):
        _ = DataHandlerAnnData(adata, **DATAHANDLER_DEFAULT_KWARGS)


def test_all_batches_have_reference_false_anndata(data_anndata: AnnData,
                                                  DATAHANDLER_DEFAULT_KWARGS: dict):  # noqa
    adata = data_anndata
    adata.obs["reference"] = adata.obs["reference"].astype(str)
    adata.obs.loc[
        adata.obs["batch"] == "3",
        "reference"
    ] = "other"
    with pytest.raises(ValueError):
        _ = DataHandlerAnnData(adata, **DATAHANDLER_DEFAULT_KWARGS)


def test_get_reference_files(metadata: pd.DataFrame,
                             INPUT_DIR: Path):
    dataset = DataHandlerFCS(metadata = metadata,
                             input_directory = INPUT_DIR)
    ref_samples_ctrl = metadata.loc[
        metadata["reference"] == "ref", "file_name"
    ].tolist()
    ref_samples_test = dataset._get_reference_file_names()
    assert all(k in ref_samples_ctrl for k in ref_samples_test)


def test_get_reference_files_anndata(data_anndata: AnnData,
                                     metadata: pd.DataFrame,
                                     DATAHANDLER_DEFAULT_KWARGS: dict):
    md = metadata
    dh = DataHandlerAnnData(data_anndata, **DATAHANDLER_DEFAULT_KWARGS)
    ref_samples_ctrl = md.loc[md["reference"] == "ref", "file_name"].tolist()
    ref_samples_test = dh._get_reference_file_names()
    assert all(k in ref_samples_ctrl for k in ref_samples_test)


def test_get_validation_files(metadata: pd.DataFrame,
                              INPUT_DIR: Path):
    dataset = DataHandlerFCS(metadata = metadata,
                             input_directory = INPUT_DIR)
    val_samples_ctrl = metadata.loc[
        metadata["reference"] != "ref", "file_name"
    ].tolist()
    val_samples_test = dataset._get_validation_file_names()

    assert all(k in val_samples_ctrl for k in val_samples_test)


def test_get_validation_files_anndata(data_anndata: AnnData,
                                      metadata: pd.DataFrame,
                                      DATAHANDLER_DEFAULT_KWARGS: dict):
    md = metadata
    dh = DataHandlerAnnData(data_anndata, **DATAHANDLER_DEFAULT_KWARGS)
    val_samples_ctrl = md.loc[md["reference"] != "ref", "file_name"].tolist()
    val_samples_test = dh._get_validation_file_names()

    assert all(k in val_samples_ctrl for k in val_samples_test)


def test_all_file_names(metadata: pd.DataFrame,
                        INPUT_DIR: Path):
    dataset = DataHandlerFCS(metadata = metadata,
                             input_directory = INPUT_DIR)
    samples = metadata.loc[:, "file_name"].tolist()

    assert all(k in samples for k in dataset._all_file_names)


def test_all_file_names_anndata(data_anndata: AnnData,
                                metadata: pd.DataFrame,
                                DATAHANDLER_DEFAULT_KWARGS: dict):
    md = metadata
    dh = DataHandlerAnnData(data_anndata, **DATAHANDLER_DEFAULT_KWARGS)
    samples = md.loc[:, "file_name"].tolist()

    assert all(k in samples for k in dh._all_file_names)


def test_correct_df_shape_all_channels(metadata: pd.DataFrame,
                                       INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata = metadata,
                        input_directory = INPUT_DIR,
                        channels = "all")
    assert dh.ref_data_df.shape == (3000, 55)


def test_correct_df_shape_all_channels_anndata(data_anndata: AnnData,
                                               DATAHANDLER_DEFAULT_KWARGS: dict):
    kwargs = DATAHANDLER_DEFAULT_KWARGS.copy()
    kwargs["channels"] = "all"
    dh = DataHandlerAnnData(data_anndata, **kwargs)
    assert dh.ref_data_df.shape == (3000, 55)


def test_correct_df_shape_markers(datahandlerfcs: DataHandlerFCS):
    # Time and Event_length are excluded
    assert datahandlerfcs.ref_data_df.shape == (3000, 53)


def test_correct_df_shape_markers_anndata(datahandleranndata: DataHandlerAnnData,
                                          DATAHANDLER_DEFAULT_KWARGS: dict):
    # Time and Event_length are excluded
    print(DATAHANDLER_DEFAULT_KWARGS)
    assert datahandleranndata.ref_data_df.shape == (3000, 53)


def test_correct_df_shape_channellist(metadata: pd.DataFrame,
                                      detectors: list[str],
                                      INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata = metadata,
                        input_directory = INPUT_DIR,
                        channels = detectors[:30])
    assert dh.ref_data_df.shape == (3000, 30)


def test_correct_df_shape_channellist_anndata(data_anndata: AnnData,
                                              detectors: list[str],
                                              DATAHANDLER_DEFAULT_KWARGS: dict):
    kwargs: dict = DATAHANDLER_DEFAULT_KWARGS.copy()
    kwargs["channels"] = detectors[:30]
    dh = DataHandlerAnnData(data_anndata, **kwargs)
    assert dh.ref_data_df.shape == (3000, 30)


def test_correct_channel_indices(metadata: pd.DataFrame,
                                 INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata = metadata,
                        input_directory = INPUT_DIR,
                        channels = "markers")
    fcs_file = dh._read_fcs_file(file_name = metadata["file_name"].tolist()[0])
    fcs_channels = fcs_file.channels.index.tolist()
    channel_idxs = dh._channel_indices
    channels_from_channel_idxs = [fcs_channels[i] for i in channel_idxs]
    assert dh.ref_data_df.columns.tolist() == channels_from_channel_idxs


def test_correct_channel_indices_anndata(data_anndata: AnnData,
                                         DATAHANDLER_DEFAULT_KWARGS: dict):
    dh = DataHandlerAnnData(data_anndata, **DATAHANDLER_DEFAULT_KWARGS)
    fcs_channels = data_anndata.var_names.tolist()
    channel_idxs = dh._channel_indices
    channels_from_channel_idxs = [fcs_channels[i] for i in channel_idxs]
    assert dh.ref_data_df.columns.tolist() == channels_from_channel_idxs


def test_correct_channel_indices_channellist(metadata: pd.DataFrame,
                                             detectors: list[str],
                                             INPUT_DIR: Path):
    dh = DataHandlerFCS(metadata = metadata,
                        input_directory = INPUT_DIR,
                        channels = detectors[:30])
    fcs_file = dh._read_fcs_file(file_name = metadata["file_name"].tolist()[0])
    fcs_channels = fcs_file.channels.index.tolist()
    channel_idxs = dh._channel_indices
    channels_from_channel_idxs = [fcs_channels[i] for i in channel_idxs]
    assert dh.ref_data_df.columns.tolist() == channels_from_channel_idxs


def test_correct_channel_indices_channellist_anndata(data_anndata: AnnData,
                                                     detectors: list[str],
                                                     DATAHANDLER_DEFAULT_KWARGS: dict):  # noqa
    kwargs: dict = DATAHANDLER_DEFAULT_KWARGS.copy()
    kwargs["channels"] = detectors[:30]
    dh = DataHandlerAnnData(data_anndata, **kwargs)
    fcs_channels = data_anndata.var_names.tolist()
    channel_idxs = dh._channel_indices
    channels_from_channel_idxs = [fcs_channels[i] for i in channel_idxs]
    assert dh.ref_data_df.columns.tolist() == channels_from_channel_idxs


def test_correct_index_of_ref_data_df(datahandlerfcs: DataHandlerFCS):
    assert isinstance(datahandlerfcs.ref_data_df.index, pd.MultiIndex)
    assert list(datahandlerfcs.ref_data_df.index.names) == ["reference",
                                                            "batch",
                                                            "file_name"]


def test_correct_index_of_ref_data_df_anndata(datahandleranndata: DataHandlerAnnData):  # noqa
    assert isinstance(datahandleranndata.ref_data_df.index, pd.MultiIndex)
    assert list(datahandleranndata.ref_data_df.index.names) == ["reference",
                                                                "batch",
                                                                "file_name"]


def test_init_metadata_columns(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    dh._init_metadata_columns(reference_column = "refff",
                              reference_value = "ref_value",
                              batch_column = "BATCHZ",
                              sample_identifier_column = "diverse")
    assert dh._reference_column == "refff"
    assert dh._reference_value == "ref_value"
    assert dh._batch_column == "BATCHZ"
    assert dh._sample_identifier_column == "diverse"


def test_append_metadata_to_df(datahandleranndata: DataHandlerAnnData,
                               metadata: pd.DataFrame):
    dh = datahandleranndata
    req_file = metadata["file_name"].tolist()[0]
    df = pd.DataFrame(
        data = np.zeros(shape = (100, 10)),
        columns = pd.Index([str(i) for i in range(10)]),
        index = list(range(100))
    )
    dh._append_metadata_to_df(df = df,
                              file_name = req_file)
    assert dh._reference_column in df.columns
    assert dh._batch_column in df.columns
    ref_value = metadata.loc[
        metadata["file_name"] == req_file,
        "reference"
    ].iloc[0]
    batch_value = metadata.loc[
        metadata["file_name"] == req_file,
        "batch"
    ].iloc[0]
    assert str(df[dh._reference_column].unique().tolist()[0]) == str(ref_value)
    assert str(df[dh._batch_column].unique().tolist()[0]) == str(batch_value)


def test_get_batch(datahandleranndata: DataHandlerAnnData,
                   metadata: pd.DataFrame):

    dh = datahandleranndata
    req_file = metadata["file_name"].tolist()[0]

    batch_value = metadata.loc[
        metadata["file_name"] == req_file,
        "batch"
    ].iloc[0]

    dh_batch_value = dh.get_batch(file_name = req_file)
    assert str(batch_value) == str(dh_batch_value)


def test_get_corresponding_reference_file(datahandleranndata: DataHandlerAnnData,  # noqa
                                          metadata: pd.DataFrame):
    dh = datahandleranndata
    req_file = metadata["file_name"].tolist()[1]
    curr_batch = dh.get_batch(req_file)
    batch_files = metadata.loc[
        metadata["batch"] == int(curr_batch),
        "file_name"
    ].tolist()
    corr_file = [file for file in batch_files if file != req_file][0]
    assert dh._find_corresponding_reference_file(req_file) == corr_file


def test_get_corresponding_ref_dataframe(datahandleranndata: DataHandlerAnnData,
                                         metadata: pd.DataFrame):
    dh = datahandleranndata
    req_file = metadata["file_name"].tolist()[1]
    df = dh.get_corresponding_ref_dataframe(req_file,
                                            raw = True,
                                            annot_file_name = True)
    file_df = dh.get_dataframe(req_file,
                               raw = True,
                               annot_file_name = True)
    assert df.shape == (1000, 55 + 1)
    assert dh._sample_identifier_column in df.columns
    assert not np.array_equal(
        np.array(df[:14].values),
        np.array(file_df[:14].values)
    )


def test_get_ref_data_df(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    assert dh.ref_data_df.equals(dh.get_ref_data_df())


def test_get_ref_data_df_subsampled(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    df = dh.get_ref_data_df_subsampled(n = 3000)
    assert df.shape[0] == 3000


def test_subsample_df(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    df = dh.ref_data_df
    assert isinstance(df, pd.DataFrame)
    df_subsampled = dh._subsample_df(df,
                                     n = 3000)
    assert df_subsampled.shape[0] == 3000


def test_find_marker_channels(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    detectors = dh._all_detectors
    markers = dh._find_marker_channels(detectors)
    technicals = dh._cytof_technicals
    assert not any(
        k in markers
        for k in technicals
    )


def test_technical_setters(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    new_list = ["some", "channels"]
    dh.flow_technicals = new_list
    assert dh.flow_technicals == ["some", "channels"]

