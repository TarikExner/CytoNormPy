import pytest
from anndata import AnnData
import pandas as pd
import numpy as np

from cytonormpy._dataset._dataset import DataHandlerAnnData


def test_missing_colname(data_anndata: AnnData,
                         DATAHANDLER_DEFAULT_KWARGS: dict):
    adata = data_anndata.copy()
    adata.obs = adata.obs.drop("reference", axis = 1)
    with pytest.raises(KeyError):
        _ = DataHandlerAnnData(adata, **DATAHANDLER_DEFAULT_KWARGS)
    adata = data_anndata.copy()
    adata.obs = adata.obs.drop("batch", axis = 1)
    with pytest.raises(KeyError):
        _ = DataHandlerAnnData(adata, **DATAHANDLER_DEFAULT_KWARGS)
    adata = data_anndata.copy()
    adata.obs = adata.obs.drop("file_name", axis = 1)
    with pytest.raises(KeyError):
        _ = DataHandlerAnnData(adata, **DATAHANDLER_DEFAULT_KWARGS)


def test_create_ref_data_df(datahandleranndata: DataHandlerAnnData):
    dh = datahandleranndata
    df = dh._create_ref_data_df()
    assert isinstance(df, pd.DataFrame)
    df = df.reset_index()
    assert all(
        k in df.columns
        for k in [dh._reference_column,
                  dh._batch_column,
                  dh._sample_identifier_column]
    )
    assert df.shape[0] == 3000


def test_condense_metadata(data_anndata: AnnData,
                           datahandleranndata: DataHandlerAnnData):
    obs = data_anndata.obs
    dh = datahandleranndata
    df = dh._condense_metadata(
        obs = obs,
        reference_column = dh._reference_column,
        batch_column = dh._batch_column,
        sample_identifier_column = dh._sample_identifier_column
    )
    assert isinstance(df, pd.DataFrame)
    assert all(
        all(df[col].duplicated() == False)  # noqa
        for col in [dh._sample_identifier_column]
    )
    assert df.shape == df.drop_duplicates().shape


def test_get_dataframe(datahandleranndata: DataHandlerAnnData,
                       metadata: pd.DataFrame):
    req_file = metadata["file_name"].tolist()[0]
    dh = datahandleranndata
    df = dh.get_dataframe(req_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 53)
    assert "file_name" not in df.columns


def test_write_anndata(datahandleranndata: DataHandlerAnnData,
                       metadata: pd.DataFrame):
    dh = datahandleranndata
    insertion_data = np.zeros(shape = (1000, dh._channel_indices.shape[0]))
    req_file = metadata["file_name"].tolist()[0]
    insertion_data = pd.DataFrame(
        data = insertion_data,
        columns = dh.channels,
        index = list(range(insertion_data.shape[0]))
    )
    dh.write(file_name = req_file,
             data = insertion_data)
    subset_adata = dh.adata[
        dh.adata.obs[dh._sample_identifier_column] == req_file,
        :
    ]
    df = subset_adata.to_df(layer = dh._key_added)
    changed = df.iloc[:, dh._channel_indices]
    assert (changed.sum(axis = 0) == 0).all()
