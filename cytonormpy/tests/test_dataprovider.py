import pytest
from cytonormpy._dataset._dataprovider import DataProviderFCS, DataProvider, DataProviderAnnData
from cytonormpy._transformation._transformations import AsinhTransformer
import pandas as pd
import numpy as np
from anndata import AnnData

from cytonormpy._dataset._metadata import Metadata


def _read_metadata_from_fixture(metadata: pd.DataFrame) -> Metadata:
    return Metadata(
        metadata=metadata,
        sample_identifier_column="file_name",
        batch_column="batch",
        reference_column="reference",
        reference_value="ref",
    )


@pytest.fixture
def PROVIDER_KWARGS_FCS(metadata: pd.DataFrame) -> dict:
    return dict(
        input_directory="some/path/",
        truncate_max_range=True,
        metadata=_read_metadata_from_fixture(metadata),
        channels=None,
        transformer=None,
    )


@pytest.fixture
def PROVIDER_KWARGS_ANNDATA(metadata: pd.DataFrame) -> dict:
    return dict(
        adata=AnnData(),
        layer="compensated",
        metadata=_read_metadata_from_fixture(metadata),
        channels=None,
        transformer=None,
    )


def test_class_hierarchy_fcs(PROVIDER_KWARGS_FCS: dict):
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    assert isinstance(x, DataProvider)


def test_class_hierarchy_anndata(PROVIDER_KWARGS_ANNDATA: dict):
    x = DataProviderAnnData(**PROVIDER_KWARGS_ANNDATA)
    assert isinstance(x, DataProvider)


def test_channels_setters(PROVIDER_KWARGS_FCS: dict):
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    assert x.channels is None
    x.channels = ["some", "channels"]
    assert x.channels == ["some", "channels"]


def test_select_channels_method_channels_equals_none(PROVIDER_KWARGS_FCS: dict):
    """if channels is None, the original data are returned"""
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    data = pd.DataFrame(
        data=np.ones(shape=(3, 3)), columns=["ch1", "ch2", "ch3"], index=list(range(3))
    )
    df = x.select_channels(data)
    assert data.equals(df)


def test_select_channels_method_channels_set(PROVIDER_KWARGS_FCS: dict):
    """if channels is a list, only the channels are kept"""
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    x.channels = ["ch1", "ch2"]
    data = pd.DataFrame(
        data=np.ones(shape=(3, 3)), columns=["ch1", "ch2", "ch3"], index=list(range(3))
    )
    df = x.select_channels(data)
    assert df.shape == (3, 2)
    assert "ch3" not in df.columns
    assert "ch1" in df.columns
    assert "ch2" in df.columns


def test_transform_method_no_transformer(PROVIDER_KWARGS_FCS: dict):
    """if transformer is None, the original data are returned"""
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    data = pd.DataFrame(
        data=np.ones(shape=(3, 3)), columns=["ch1", "ch2", "ch3"], index=list(range(3))
    )
    df = x.transform_data(data)
    assert data.equals(df)


def test_transform_method_with_transformer(PROVIDER_KWARGS_FCS: dict):
    """if channels is None, the original data are returned"""
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    x.transformer = AsinhTransformer()
    data = pd.DataFrame(
        data=np.ones(shape=(3, 3)), columns=["ch1", "ch2", "ch3"], index=list(range(3))
    )
    df = x.transform_data(data)
    assert all(df == np.arcsinh(1 / 5))
    assert all(df.columns == data.columns)
    assert all(df.index == data.index)


def test_inv_transform_method_no_transformer(PROVIDER_KWARGS_FCS: dict):
    """if transformer is None, the original data are returned"""
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    data = pd.DataFrame(
        data=np.ones(shape=(3, 3)), columns=["ch1", "ch2", "ch3"], index=list(range(3))
    )
    df = x.inverse_transform_data(data)
    assert data.equals(df)


def test_inv_transform_method_with_transformer(PROVIDER_KWARGS_FCS: dict):
    """if channels is None, the original data are returned"""
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    x.transformer = AsinhTransformer()
    data = pd.DataFrame(
        data=np.ones(shape=(3, 3)), columns=["ch1", "ch2", "ch3"], index=list(range(3))
    )
    df = x.transform_data(data)
    assert all(df == np.sinh(1) * 5)
    assert all(df.columns == data.columns)
    assert all(df.index == data.index)


def test_annotate_metadata(metadata: pd.DataFrame, PROVIDER_KWARGS_FCS: dict):
    x = DataProviderFCS(**PROVIDER_KWARGS_FCS)
    data = pd.DataFrame(
        data=np.ones(shape=(3, 3)), columns=["ch1", "ch2", "ch3"], index=list(range(3))
    )
    file_name = metadata["file_name"].tolist()[0]
    df = x.annotate_metadata(data, file_name)
    assert all(
        k in df.index.names
        for k in [
            x.metadata.sample_identifier_column,
            x.metadata.reference_column,
            x.metadata.batch_column,
        ]
    )
