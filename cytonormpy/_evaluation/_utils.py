import pandas as pd
from os import PathLike

from typing import Union, Optional
from anndata import AnnData

from .._dataset._dataprovider import DataProviderFCS, DataProviderAnnData
from .._transformation import Transformer

def _prepare_data_fcs(input_directory: PathLike,
                      files: Union[list[str], str],
                      channels: Optional[Union[list[str], pd.Index]],
                      cell_labels: Optional[dict] = None,
                      truncate_max_range: bool = False,
                      transformer: Optional[Transformer] = None
                      ) -> tuple[pd.DataFrame, Union[list[str], pd.Index]]:

    df = _parse_fcs_dfs(
        input_directory = input_directory,
        file_list = files,
        cell_labels = cell_labels,
        channels = channels,
        truncate_max_range = truncate_max_range,
        transformer = transformer
    )

    df = df.set_index(["file_name", "label"])

    if channels is None:
        channels = df.columns.tolist()
        assert channels is not None

    return df, channels

def _prepare_data_anndata(adata: AnnData,
                          file_list: Union[list[str], str],
                          channels: Optional[Union[list[str], pd.Index]],
                          layer: str,
                          sample_identifier_column: str = "file_name",
                          cell_labels: Optional[str] = None,
                          transformer: Optional[Transformer] = None
                          ) -> tuple[pd.DataFrame, Union[list[str], pd.Index]]:

    df = _parse_anndata_dfs(
        adata = adata,
        file_list = file_list,
        layer = layer,
        cell_labels = cell_labels,
        sample_identifier_column = sample_identifier_column,
        channels = channels,
        transformer = transformer
    )

    df = df.set_index([sample_identifier_column, "label"])

    if channels is None:
        channels = df.columns.tolist()
        assert channels is not None

    return df, channels

def _parse_anndata_dfs(adata: AnnData,
                       file_list: Union[list[str], str],
                       layer: str,
                       sample_identifier_column,
                       cell_labels: Optional[str],
                       transformer: Optional[Transformer],
                       channels: Optional[list[str]] = None):
    provider = DataProviderAnnData(
        adata = adata,
        layer = layer,
        sample_identifier_column = sample_identifier_column,
        channels = channels,
        transformer = transformer
    )
    df = provider.parse_anndata_df(file_list)
    df = provider.select_channels(df)
    df = provider.transform_data(df)
    df[sample_identifier_column] = adata.obs.loc[
        adata.obs[sample_identifier_column].isin(file_list),
        sample_identifier_column
    ].tolist()
    if cell_labels is not None:
        df["label"] = adata.obs.loc[
            adata.obs[sample_identifier_column].isin(file_list),
            cell_labels
        ].tolist()
    else:
        df["label"] = "all_cells"

    return df
    
def _parse_fcs_dfs(input_directory,
                   file_list: Union[list[str], str],
                   channels: Optional[list[str]] = None,
                   cell_labels: Optional[dict] = None,
                   truncate_max_range: bool = False,
                   transformer: Optional[Transformer] = None) -> pd.DataFrame:

    provider = DataProviderFCS(
        input_directory = input_directory,
        truncate_max_range = truncate_max_range,
        sample_identifier_column = "file_name",
        channels = channels,
        transformer = transformer
    )
    dfs = []
    for file in file_list:
        data = provider._reader.parse_fcs_df(file)
        data = provider.select_channels(data)
        data = provider.transform_data(data)
        data = provider._annotate_sample_identifier(data, file)
        if cell_labels is not None:
            data["label"] = cell_labels[file]
        else:
            data["label"] = "all_cells"
        dfs.append(data)

    return pd.concat(dfs, axis = 0)

def _annotate_origin(df: pd.DataFrame,
                     origin: str) -> pd.DataFrame:
    """\
    Annotates the origin of the data and sets the index.
    """
    df["origin"] = origin
    df = df.set_index("origin", append = True, drop = True)
    return df
