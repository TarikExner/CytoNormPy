import pandas as pd
from os import PathLike
from scipy.stats import median_abs_deviation

from .._dataset._fcs_file import FCSFile
from .._transformation import Transformer

from typing import Union, Optional


def _mad_evaluation(df: pd.DataFrame,
                    channels: Union[list[str], str],
                    groupby: Optional[Union[list[str], str]] = None,
                    sample_identifier: str = "file_name",
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
    sample_identifier
        Specifies the column in `df` that is unique to the samples.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per
    `sample_identifier` or per `sample_identifier` and `groupby`

    """
    def _mad(group, columns):
        return group[columns].apply(
            lambda x: median_abs_deviation(
                x,
                scale = "normal"
            ), axis = 0
        )
    
    if groupby is None:
        _groupby = None
        groupby = "cell_type"
        df[groupby] = "all_cells"
    else:
        _groupby = groupby
        
    if not isinstance(groupby, list):
        groupby = [groupby]

    groupings = [sample_identifier] + groupby

    all_cells = df.groupby([sample_identifier]).apply(lambda x: _mad(x, channels))
    
    all_cells[groupby] = "all_cells"
    all_cells = all_cells.set_index(groupby, append = True, drop = True)
    
    if _groupby is not None:
        groupby_res = df.groupby(groupings).apply(lambda x: _mad(x, channels))
        res = pd.concat([all_cells, groupby_res], axis = 0)
    else:
        res = all_cells

    return res


def evaluate_mad(input_directory: PathLike,
                 files: Union[list[str], str],
                 channels: Optional[list[str]] = None,
                 cell_labels: Optional[dict] = None,
                 transformer: Optional[Transformer] = None
                 ):
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
    cell labels
        A dictionary of shape {file_name: [cell_label, cell_label, ...]}.
        The cell labels will be added to the dataframe. If None,
        MADs will be calculated per file and channel.
    channels:
        A list of detectors to analyze.
    transformer
        An instance of the cytonormpy transformers.

    Returns
    -------
    A :class:`pandas.DataFrame` containing the MAD values per
    file or per file and `cell_label`.

    """

    if not isinstance(files, list):
        files = [files]

    dfs = []

    for file in files:
        fcs = FCSFile(input_directory = input_directory,
                      file_name = file,
                      truncate_max_range = False)
        df = fcs.to_df()
        if cell_labels is not None:
            df["label"] = cell_labels[file]
        df["file_name"] = file
        dfs.append(df)
    df = pd.concat(dfs, axis = 0)
    if channels is None:
        channels = [
            col
            for col in df
            if not any(
                k in col
                for k in ["label", "file_name"]
            )
        ]

    if transformer is not None:
        df[channels] = transformer.transform(
            df[channels].values
        )

    return _mad_evaluation(
        df,
        channels = channels,
        groupby = "label" if cell_labels is not None else None
    )



