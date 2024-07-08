import pandas as pd
from os import PathLike
from scipy.stats import median_abs_deviation

from .._transformation import Transformer
from .._dataset._dataprovider import DataProviderFCS, DataProviderAnnData

from typing import Union, Optional
from anndata import AnnData

ALLOWED_GROUPINGS_FCS = [
    "file_name",
    ["file_name"],
    "label",
    ["label"],
    ["file_name", "label"],
    ["label", "file_name"]
]

def _calculate_mads_per_frame(df: pd.DataFrame,
                              channels: Union[list[str], pd.Index],
                              groupby: list[str]) -> pd.DataFrame:
    if "file_name" in groupby:
        all_cells = _mad_per_group(
            df,
            channels = channels,
            groupby = ["file_name"]
        )
        all_cells["label"] = "all_cells"
        all_cells = all_cells.set_index("label", append = True, drop = True)

        unique_label_levels = df.index.get_level_values("label").unique().tolist()
        
        if groupby == ["file_name"] or len(unique_label_levels) == 1:
            return all_cells
        else:
            grouped = _mad_per_group(
                df,
                channels = channels,
                groupby = groupby
            )
            return pd.concat([all_cells, grouped], axis = 0)
    else:
        return _mad_per_group(
            df,
            channels = channels,
            groupby = groupby
        )

def _mad_per_group(df: pd.DataFrame,
                   channels: Union[list[str], pd.Index],
                   groupby: list[str]
                   ) -> pd.DataFrame:
    """\
    Function to evaluate the Median Absolute Deviation on a dataframe.
    This function is not really meant to be used from outside, but
    rather embedded into higher level functions.

    Parameters
    ----------
    channels
        A list of detectors present in the dataframe.
    groupby
        Specifies an additional axis to be grouped by. By default,
        the dataframe will be grouped by `sample_identifier`, which
        defaults to 'file_name'. Can be a list of columns or a single
        column.
    channels:
        A list of detectors to analyze.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per group.

    """

    def _mad(group, columns):
        return group[columns].apply(
            lambda x: median_abs_deviation(
                x,
                scale = "normal"
            ), axis = 0
        )

    return df.groupby(groupby).apply(lambda x: _mad(x, channels))

def _annotate_origin(df: pd.DataFrame,
                     origin: str) -> pd.DataFrame:
    """\
    Annotates the origin of the data and sets the index.
    """
    df["origin"] = origin
    df = df.set_index("origin", append = True, drop = True)
    return df

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

def mad_comparison_from_anndata(adata: AnnData,
                                file_list: Union[list[str], str],
                                channels: Optional[list[str]],
                                orig_layer: str,
                                norm_layer: str,
                                sample_identifier_column: str = "file_name",
                                cell_labels: Optional[str] = None,
                                groupby: Optional[Union[list[str], str]] = None,
                                transformer: Optional[Transformer] = None) -> pd.DataFrame:
    """
    This function is a wrapper around `mad_from_anndata` that directly combines the
    normalized and unnormalized dataframes. Currently only works if the
    normalized and unnormalized files are in the same directory.

    Parameters
    ----------
    adata
        The AnnData object
    file_list
        A list of files. Used in conjunction with `sample_identifier_column`.
    channels:
        A list of detectors to analyze.
    orig_layer
        The layer where the original data are stored.
    norm_layer
        The layer where the normalized data are stored.
    sample_identifier_column
        Specifies the column in `adata.obs` in which the samples are identified.
    cell labels
        Specifies the column in `adata.obs` containing cell labels.
        The cell labels will be added to the dataframe. If None,
        MADs will be calculated per file and channel.
    groupby
        Specify on what the MADs should be grouped. Can be "file_name",
        "label" or ["file_name", "label"].
    transformer
        An instance of the cytonormpy transformers.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per file or per file and `cell_label`.

    """
    kwargs = locals()
    orig_layer = kwargs.pop("orig_layer")
    norm_layer = kwargs.pop("norm_layer")
    orig_df = mad_from_anndata(
        origin = "unnormalized",
        layer = orig_layer,
        **kwargs
    )
    norm_df = mad_from_anndata(
        origin = "normalized",
        layer = norm_layer,
        **kwargs
    )

    return pd.concat([orig_df, norm_df], axis = 0)


def mad_from_anndata(adata: AnnData,
                     file_list: Union[list[str], str],
                     channels: Optional[list[str]],
                     layer: str,
                     sample_identifier_column: str = "file_name",
                     cell_labels: Optional[str] = None,
                     groupby: Optional[Union[list[str], str]] = None,
                     origin: Optional[str] = None,
                     transformer: Optional[Transformer] = None) -> pd.DataFrame:
    """\
    Function to evaluate the MAD on an AnnData file.

    Parameters
    ----------
    adata
        The AnnData object
    file_list
        A list of files. Used in conjunction with `sample_identifier_column`.
    channels:
        A list of detectors to analyze.
    layer
        The layer where the data are stored.
    sample_identifier_column
        Specifies the column in `adata.obs` in which the samples are identified.
    cell labels
        Specifies the column in `adata.obs` containing cell labels.
        The cell labels will be added to the dataframe. If None,
        MADs will be calculated per file and channel.
    groupby
        Specify on what the MADs should be grouped. Can be "file_name",
        "label" or ["file_name", "label"].
    origin
        Annotates the files with their origin, e.g. 'original' or 'normalized'.
    transformer
        An instance of the cytonormpy transformers.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per file or per file and `cell_label`.

    """

    if groupby is None:
        groupby = sample_identifier_column
    
    if not isinstance(groupby, list):
        groupby = [groupby]
    
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

    df = _calculate_mads_per_frame(
        df, channels, groupby
    )
    
    if origin is not None:
        df = _annotate_origin(df, origin)

    return df

def mad_comparison_from_fcs(input_directory: PathLike,
                            original_files: Union[list[str], str],
                            normalized_files: Union[list[str], str],
                            norm_prefix: str = "Norm_",
                            channels: Optional[list[str]] = None,
                            cell_labels: Optional[dict] = None,
                            groupby: Optional[Union[list[str], str]] = None,
                            truncate_max_range: bool = False,
                            transformer: Optional[Transformer] = None) -> pd.DataFrame:
    """
    This function is a wrapper around `mad_from_fcs` that directly combines the
    normalized and unnormalized dataframes. Currently only works if the
    normalized and unnormalized files are in the same directory.

    Parameters
    ----------
    input_directory
        Path specifying the input directory in which the
        .fcs files are stored. If left None, the current
        working directory is assumed.
    original_files
        A list of original files ending with .fcs.
    normalized_files
        A list of normalized files ending with .fcs.
    norm_prefix
        The prefix of the normalized files.
    channels:
        A list of detectors to analyze.
    cell labels
        A dictionary of shape {file_name: [cell_label, cell_label, ...]}.
        The cell labels will be added to the dataframe. If None,
        MADs will be calculated per file and channel.
    groupby
        Specify on what the MADs should be grouped. Can be "file_name",
        "label" or ["file_name", "label"].
    truncate_max_range
        If `True`, FCS data will be truncated to the range specified
        in the PnR values of the file.
    transformer
        An instance of the cytonormpy transformers.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per file or per file and `cell_label`.

    """
    kwargs = locals()
    orig_files = kwargs.pop("original_files")
    norm_files = kwargs.pop("normalized_files")
    norm_prefix = kwargs.pop("norm_prefix")
    orig_df = mad_from_fcs(
        origin = "unnormalized",
        files = orig_files,
        **kwargs
    )
    norm_df = mad_from_fcs(
        origin = "normalized",
        files = norm_files,
        **kwargs
    )

    # we have to rename the file_names
    df = pd.concat([orig_df, norm_df], axis = 0)

    df = df.reset_index(level = "file_name")
    if "file_name" in df.index.names:
        df["file_name"] = [
            entry.strip(norm_prefix + "_")
            for entry in df["file_name"].tolist()
        ]
        df = df.set_index("file_name", append = True, drop = True)

    return df
                     
def mad_from_fcs(input_directory: PathLike,
                 files: Union[list[str], str],
                 channels: Optional[list[str]] = None,
                 cell_labels: Optional[dict] = None,
                 groupby: Optional[Union[list[str], str]] = None,
                 truncate_max_range: bool = False,
                 origin: Optional[str] = None,
                 transformer: Optional[Transformer] = None) -> pd.DataFrame:
    """\
    Function to evaluate the MAD on a given list of FCS-files.

    Parameters
    ----------
    input_directory
        Path specifying the input directory in which the
        .fcs files are stored. If left None, the current
        working directory is assumed.
    files
        A list of files ending with .fcs.
    channels:
        A list of detectors to analyze.
    cell labels
        A dictionary of shape {file_name: [cell_label, cell_label, ...]}.
        The cell labels will be added to the dataframe. If None,
        MADs will be calculated per file and channel.
    groupby
        Specify on what the MADs should be grouped. Can be "file_name",
        "label" or ["file_name", "label"].
    truncate_max_range
        If `True`, FCS data will be truncated to the range specified
        in the PnR values of the file.
    origin
        Annotates the files with their origin, e.g. 'original' or 'normalized'.
    transformer
        An instance of the cytonormpy transformers.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per file or per file and `cell_label`.

    """
    if not isinstance(files, list):
        files = [files]

    if groupby is None:
        groupby = "file_name"
    
    if groupby not in ALLOWED_GROUPINGS_FCS:
        raise ValueError(
            f"Groupby has to be one of {ALLOWED_GROUPINGS_FCS} " +
            f"but was {groupby}."
        )

    if not isinstance(groupby, list):
        groupby = [groupby]

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

    df = _calculate_mads_per_frame(
        df, channels, groupby
    )
    
    if origin is not None:
        df = _annotate_origin(df, origin)

    return df
