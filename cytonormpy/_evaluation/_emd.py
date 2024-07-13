import pandas as pd
from os import PathLike

from typing import Union, Optional
from anndata import AnnData

from .._transformation import Transformer
from ._emd_utils import _calculate_emd_per_frame
from ._utils import (_annotate_origin,
                     _prepare_data_fcs,
                     _prepare_data_anndata)


def emd_comparison_from_anndata(adata: AnnData,
                                file_list: Union[list[str], str],
                                channels: Optional[list[str]],
                                orig_layer: str,
                                norm_layer: str,
                                sample_identifier_column: str = "file_name",
                                cell_labels: Optional[str] = None,
                                transformer: Optional[Transformer] = None) -> pd.DataFrame:
    """
    This function is a wrapper around `emd_from_anndata` that directly combines the
    normalized and unnormalized dataframes. 

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
    transformer
        An instance of the cytonormpy transformers.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per file or per file and `cell_label`.

    """
    kwargs = locals()
    orig_layer = kwargs.pop("orig_layer")
    norm_layer = kwargs.pop("norm_layer")
    orig_df = emd_from_anndata(
        origin = "unnormalized",
        layer = orig_layer,
        **kwargs
    )
    norm_df = emd_from_anndata(
        origin = "normalized",
        layer = norm_layer,
        **kwargs
    )

    return pd.concat([orig_df, norm_df], axis = 0)


def emd_from_anndata(adata: AnnData,
                     file_list: Union[list[str], str],
                     channels: Optional[list[str]],
                     layer: str,
                     sample_identifier_column: str = "file_name",
                     cell_labels: Optional[str] = None,
                     origin: Optional[str] = None,
                     transformer: Optional[Transformer] = None) -> pd.DataFrame:
    """\
    Function to evaluate the EMD on an AnnData file.

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
    
    df, channels = _prepare_data_anndata(
        adata = adata,
        file_list = file_list,
        layer = layer,
        cell_labels = cell_labels,
        sample_identifier_column = sample_identifier_column,
        channels = channels,
        transformer = transformer
    )


    df = _calculate_emd_per_frame(
        df, channels
    )
    
    if origin is not None:
        df = _annotate_origin(df, origin)

    return df

def emd_comparison_from_fcs(input_directory: PathLike,
                            original_files: Union[list[str], str],
                            normalized_files: Union[list[str], str],
                            norm_prefix: str = "Norm_",
                            channels: Optional[list[str]] = None,
                            cell_labels: Optional[dict] = None,
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
    orig_df = emd_from_fcs(
        origin = "original",
        files = orig_files,
        **kwargs
    )
    norm_df = emd_from_fcs(
        origin = "normalized",
        files = norm_files,
        **kwargs
    )

    # we have to rename the file_names
    df = pd.concat([orig_df, norm_df], axis = 0)

    return df
                     
def emd_from_fcs(input_directory: PathLike,
                 files: Union[list[str], str],
                 channels: Optional[list[str]] = None,
                 cell_labels: Optional[dict] = None,
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

    df, channels = _prepare_data_fcs(
        input_directory = input_directory,
        files = files,
        channels = channels,
        cell_labels = cell_labels,
        truncate_max_range = truncate_max_range,
        transformer = transformer
    )

    df = _calculate_emd_per_frame(
        df, channels
    )
    
    if origin is not None:
        df = _annotate_origin(df, origin)

    return df
