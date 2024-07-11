import pandas as pd
import numpy as np
import warnings
from scipy.stats import wasserstein_distance
from itertools import combinations

from typing import Union, Iterable

def _bin_array(values: list[float],
               hist_min: float,
               hist_max: float,
               bin_size: float) -> tuple[Iterable, np.ndarray]:
    """
    Bins the input arrays into bins with a size of 0.1.

    Parameters
    ----------
    values
        A list containing the input values.
    hist_min
        The minimum value of the histogram binning
    hist_max
        The maximum value of the histogram binning
    bin_size
        The binning size of the histogram binning.

    Returns
    -------
    A tuple with one Iterator containing the binnings
    and an array with normalized counts per bin.

    Notes
    -----
    The start and end of the histogram data are
    determined by the two groups that are compared.
    The min/max/bin_size calculation is therefore done
    in the function _calculate_wasserstein_distance.

    """
    bins = np.arange(
        hist_min,
        hist_max,
        bin_size
    ) + 0.0000001 # n bins, the 0.0000001 is to avoid the left edge being included in the bin
    counts, _ = np.histogram(values, bins = bins)
    
    return range(bins.shape[0] - 1), counts/sum(counts)

def _calculate_wasserstein_distance(group_pair: tuple[list[float], ...]) -> float:
    """
    Calculates the EM/Wasserstein distance.
    First, the global min and max of the two groups
    are calculated in order to setup the binning
    correctly for the respective data range.

    Parameters
    ----------
    group_pair
        A tuple containing the two value lists to be compared

    Returns
    -------
    The EMD of the two groups.

    """

    group1_max = max(group_pair[0])
    group2_max = max(group_pair[1])
    group1_min = min(group_pair[0])
    group2_min = min(group_pair[1])

    global_min = min(group1_min, group2_min)
    global_max = max(group1_max, group2_max)

    # we check for the data range in order to confer CytoNorm standards.
    # In any case: We issue a warning that the data do not look transformed.
    # The cutoff could probably set to something more sensitive, like 20.
    if global_min < -100 or global_max > 100:
        warning_msg = "The data do not look transformed. "
        warning_msg += "The output might not be meaningful"
        warnings.warn(warning_msg, UserWarning)

    # Since we subtract/add 1 later, we set it to -99 and 99, respectively.
    if global_min > -100:
        global_min = -99
    if global_max < 100:
        global_max = 99

    bin_size = _calculate_bin_size(global_min, global_max)

    u_values, u_weights = _bin_array(
        group_pair[0],
        hist_min = global_min - 1, # we extend slightly to cover all bins
        hist_max = global_max + 1, # we extend slightly to cover all bins
        bin_size = bin_size
    )
    v_values, v_weights = _bin_array(
        group_pair[1],
        hist_min = global_min - 1,
        hist_max = global_max + 1,
        bin_size = bin_size
    )

    emd = wasserstein_distance(u_values, v_values, u_weights, v_weights)

    if bin_size != 0.1:
        emd = emd * (bin_size / 0.1)

    return emd

def _calculate_bin_size(global_min: float,
                        global_max: float) -> float:
    """
    Calculates the necessary bin size. If the data range is large,
    choosing the default value of bin_size = 0.1 might lead to
    dramatic increases in computation time. Therefore, the adequate
    bin_size is calculated and the EMD value is adjusted later on.
    We suppose that a data range of 1000 gets a 0.1 bin_size. Ranges
    of 10_000 get a bin_size of 1 and so forth.

    Parameters
    ----------
    global_min
        The minimum of the data distribution
    global_max
        The maximum of the data distribution

    Returns
    -------
    The correct bin size. The minimum is set to 0.1.

    """
    diff = global_max - global_min
    adj_factor = np.ceil(np.log10(diff))
    return max(0.1, 0.0001 * 10 ** adj_factor)


def _calculate_wasserstein_distances(grouped_data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
    """
    Wrapper function in order to select the groups to be
    compared and to pass the respective data arrays to
    the actual EMD calculation.

    Parameters
    ----------
    grouped_data
        The pandas DataFrame that is correctly grouped.

    Returns
    -------
    A pandas Series containing the EMDs between all
    groups.

    """
    group_pairs = list(combinations(grouped_data, 2))
    wasserstein_dists = pd.Series(group_pairs).apply(_calculate_wasserstein_distance)
    return wasserstein_dists

def _wasserstein_per_label(label_group, channels) -> pd.Series:
    """
    Wrapper function in order to coordinate the EMD calculations.


    """

    max_dists = {}
    for channel in channels:
        grouped_data = label_group.groupby("file_name")[channel].apply(list)
        dists = _calculate_wasserstein_distances(grouped_data)
        max_dists[channel] = dists.max() if not dists.empty else float("nan")
    return pd.Series(max_dists)

def _calculate_emd_per_frame(df: pd.DataFrame,
                             channels: Union[list[str], pd.Index]) -> pd.DataFrame:

    assert all(level in df.index.names for level in ["file_name", "label"])
    n_labels = df.index.get_level_values("label").nunique()

    res = df.groupby("label").apply(
        lambda label_group: _wasserstein_per_label(label_group, channels)
    )
    if n_labels > 1:
        df = df.reset_index(level = "label")
        df["label"] = "all_cells"
        df = df.set_index("label", append = True, drop = True)
        all_cells = df.groupby("label").apply(
            lambda label_group: _wasserstein_per_label(label_group, channels)
        )

        res = pd.concat([all_cells, res], axis = 0)

    return res
