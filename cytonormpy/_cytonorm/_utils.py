import pandas as pd

class ClusterCVWarning(Warning):

    def __init__(self,
                 message):
        self.message = message

    def __str__(self):
        return repr(self.message)

def _all_cvs_below_cutoff(df: pd.DataFrame,
                          cluster_key: str,
                          sample_key: str,
                          cv_cutoff: float) -> bool:
    """\
    Calculates the CVs of sample_ID percentages per cluster.
    Then, tests if any of the CVs are larger than the cutoff.
    """
    df = df.reset_index()
    cluster_data = df[[sample_key, cluster_key]]
    assert isinstance(cluster_data, pd.DataFrame)

    cvs = _calculate_cluster_cv(df = cluster_data,
                                cluster_key = cluster_key,
                                sample_key = sample_key)
    if any([cv > cv_cutoff for cv in cvs]):
        return False
    return True


def _calculate_cluster_cv(df: pd.DataFrame,
                          cluster_key: str,
                          sample_key) -> list[float]:
    """
    Implements the testCV function of the original CytoNorm package.
    First, we determine the percentage of cells per sample in a given
    cluster. The CV is then calculated as the SD divided by the mean
    for each cluster.

    Returns
    -------
    A list of sample_ID percentage CV per cluster.

    """
    value_counts = df.groupby(cluster_key,
                              observed = True).value_counts([sample_key])
    sample_sizes = df.groupby(sample_key,
                              observed = True).size()
    percentages = pd.DataFrame(value_counts / sample_sizes, columns = ["perc"])
    cluster_by_sample = percentages.pivot_table(values = "perc",
                                                index = sample_key,
                                                columns = cluster_key)
    return list(cluster_by_sample.std() / cluster_by_sample.mean())
