import pytest
from cytonormpy._dataset._dataprovider import DataProviderFCS, DataProvider, DataProviderAnnData
from cytonormpy._transformation._transformations import AsinhTransformer
from pathlib import Path
import pandas as pd
import numpy as np
from anndata import AnnData

def _read_metadata_from_fixture(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata

provider_kwargs_fcs = dict(
    input_directory = Path("some/path/"),
    truncate_max_range = True,
    sample_identifier_column = "file_name",
    reference_column = "reference",
    batch_column = "batch",
    metadata = _read_metadata_from_fixture,
    channels = None,
    transformer = None
)

provider_kwargs_anndata = dict(
    adata = AnnData(),
    layer = "compensated",
    sample_identifier_column = "file_name",
    reference_column = "reference",
    batch_column = "batch",
    metadata = _read_metadata_from_fixture,
    channels = None,
    transformer = None
)

def test_class_hierarchy_fcs():
    x = DataProviderFCS(**provider_kwargs_fcs)
    assert isinstance(x, DataProvider)

def test_class_hierarchy_anndata():
    x = DataProviderAnnData(**provider_kwargs_anndata)
    assert isinstance(x, DataProvider)

def test_channels_setters():
    x = DataProviderFCS(**provider_kwargs_fcs)
    assert x.channels is None
    x.channels = ["some", "channels"]
    assert x.channels == ["some", "channels"]

def test_select_channels_method_channels_equals_none():
    """if channels is None, the original data are returned"""
    x = DataProviderFCS(**provider_kwargs_fcs)
    data = pd.DataFrame(
        data = np.ones(shape = (3,3)),
        columns = ["ch1", "ch2", "ch3"],
        index = list(range(3))
    )
    df = x.select_channels(data)
    assert data.equals(df)


def test_select_channels_method_channels_set():
    """if channels is a list, only the channels are kept"""
    x = DataProviderFCS(**provider_kwargs_fcs)
    x.channels = ["ch1", "ch2"]
    data = pd.DataFrame(
        data = np.ones(shape = (3,3)),
        columns = ["ch1", "ch2", "ch3"],
        index = list(range(3))
    )
    df = x.select_channels(data)
    assert df.shape == (3,2)
    assert "ch3" not in df.columns
    assert "ch1" in df.columns
    assert "ch2" in df.columns

def test_transform_method_no_transformer():
    """if transformer is None, the original data are returned"""
    x = DataProviderFCS(**provider_kwargs_fcs)
    data = pd.DataFrame(
        data = np.ones(shape = (3,3)),
        columns = ["ch1", "ch2", "ch3"],
        index = list(range(3))
    )
    df = x.transform_data(data)
    assert data.equals(df)

def test_transform_method_with_transformer():
    """if channels is None, the original data are returned"""
    x = DataProviderFCS(**provider_kwargs_fcs)
    x.transformer = AsinhTransformer()
    data = pd.DataFrame(
        data = np.ones(shape = (3,3)),
        columns = ["ch1", "ch2", "ch3"],
        index = list(range(3))
    )
    df = x.transform_data(data)
    assert all(df == np.arcsinh(1/5))
    assert all(df.columns == data.columns)
    assert all(df.index == data.index)

def test_inv_transform_method_no_transformer():
    """if transformer is None, the original data are returned"""
    x = DataProviderFCS(**provider_kwargs_fcs)
    data = pd.DataFrame(
        data = np.ones(shape = (3,3)),
        columns = ["ch1", "ch2", "ch3"],
        index = list(range(3))
    )
    df = x.inverse_transform_data(data)
    assert data.equals(df)

def test_inv_transform_method_with_transformer():
    """if channels is None, the original data are returned"""
    x = DataProviderFCS(**provider_kwargs_fcs)
    x.transformer = AsinhTransformer()
    data = pd.DataFrame(
        data = np.ones(shape = (3,3)),
        columns = ["ch1", "ch2", "ch3"],
        index = list(range(3))
    )
    df = x.transform_data(data)
    assert all(df == np.sinh(1)*5)
    assert all(df.columns == data.columns)
    assert all(df.index == data.index)

def test_annotate_metadata(metadata: pd.DataFrame):
    provider_kwargs_fcs["metadata"] = metadata
    x = DataProviderFCS(**provider_kwargs_fcs)
    data = pd.DataFrame(
        data = np.ones(shape = (3,3)),
        columns = ["ch1", "ch2", "ch3"],
        index = list(range(3))
    )
    file_name = metadata["file_name"].tolist()[0]
    df = x.annotate_metadata(data, file_name)
    assert all(
        k in df.index.names
        for k in [x._sample_identifier_column,
                  x._reference_column,
                  x._batch_column]
    )
