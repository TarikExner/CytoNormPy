import pandas as pd
from scipy.stats import median_abs_deviation

from typing import Union

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
