import pandas as pd
from typing import Union, Optional, Literal
from os import PathLike
import numpy as np
from anndata import AnnData
import pickle
import warnings

import concurrent.futures as cf

from ._utils import _all_cvs_below_cutoff, ClusterCVWarning

from .._dataset._dataset import (DataHandlerFCS,
                                 DataHandler,
                                 DataHandlerAnnData)

from .._transformation._transformations import Transformer

from .._normalization._spline_calc import (Spline,
                                           Splines,
                                           IdentitySpline)

from .._normalization._quantile_calc import (ExpressionQuantiles,
                                             GoalDistribution)

from .._clustering._cluster_algorithms import ClusterBase


class CytoNorm:
    """\
    Cytometry data are divided into batches. Each batch contains
    one or more reference files that were measured in all batches.

    First, data are setup where either FCS files can be processed
    or data stored in an anndata object. The data can be transformed
    using common cytomety transformations.

    The reference data are then clustered using the FlowSOM
    algorithm. The SOM is stored in order to predict the corresponding
    clusters for the other samples.

    For each cluster and batch, the expression values of the reference
    files at user-defined quantiles are calculated. The goal
    distribution is calculated by either calculating the mean expression
    for each cluster over all batches or by choosing one batch as
    the goal.

    Next, interpolating functions (spline functions) are calculated
    for each batch and cluster to the goal distribution. All samples
    are then transformed by calculating the output of the respective
    spline function when using the expression values as an input.

    Example
    -------

    >>> import cytonormpy as cnp
    >>>
    >>> cn = CytoNorm()
    >>>
    >>> transformer = cnp.AsinhTransformer(cofactors = 5)
    >>> cn.add_transformer(transformer)
    >>>
    >>> clusterer = cnp.FlowSOM(**flowsom_kwargs)
    >>> cn.add_clusterer(clusterer)
    >>>
    >>> cn.run_fcs_data_setup("metadata.csv")
    >>>
    >>> # equivalently for the use of anndata:
    >>> cn.run_anndata_setup(adata)
    >>>
    >>> cn.run_clustering(n_cells = 6000,
    ...                   test_cluster_cv = True)
    >>>
    >>> cn.calculate_quantiles()
    >>> cn.calculate_splines()
    >>>
    >>> cn.normalize_data()


    """

    def __init__(self) -> None:
        self._transformer = None
        self._clustering = None

    def run_fcs_data_setup(self,
                           metadata: Union[pd.DataFrame, PathLike],
                           input_directory: PathLike,
                           reference_column: str = "reference",
                           reference_value: str = "ref",
                           batch_column: str = "batch",
                           sample_identifier_column: str = "file_name",
                           channels: Union[list[str], str, Literal["all", "markers"]] = "markers",  # noqa
                           truncate_max_range: bool = True,
                           output_directory: Optional[PathLike] = None,
                           prefix: str = "Norm"
                           ) -> None:
        """\
        Method to setup the data handling for FCS data. Will instantiate a
        :class:`~cytonormpy.DataHandlerFCS` object.

        Parameters
        ----------
        metadata
            A table containing the file names, the `batch` and
            the `reference` information. Expects the columns
            `file_name`, `batch` and `reference` where reference
            must contain `ref` for reference samples and `other`
            for non-reference samples. Can be provided as a
            :class:`~pandas.DataFrame` or a path.
        input_directory
            Path specifying the input directory in which the
            .fcs files are stored. If left None, the current
            working directory is assumed.
        reference_column
            The column in the metadata that specifies whether a sample
            is used for reference and is therefore present in all batches.
            Defaults to 'reference'.
        reference_value
            Specifies the value that is considered a reference. Defaults to
            'ref'.
        batch_column
            The column in the metadata that specifies the batch. Defaults
            to 'batch'.
        sample_identifier_column
            Specifies the column in the metadata that is unique to the samples.
            Defaults to 'file_name'.
        channels
            Can be a list of detectors (e.g. BV421-A), a single
            channel or 'all' or 'markers'. If `markers`, channels
            containing 'FSC', 'SSC', 'Time', 'AF' and CyTOF technicals
            will be excluded.
        output_directory
            Path specifying the output directory in which the
            .fcs files are saved to. If left None, the current
            input directory is assumed.
        prefix
            The prefix that are prepended to the file names
            of the normalized fcs files.

        Returns
        -------
        None, appends `._datahandler` attribute.

        """
        self._datahandler: DataHandler = DataHandlerFCS(
            metadata = metadata,
            input_directory = input_directory,
            channels = channels,
            reference_column = reference_column,
            reference_value = reference_value,
            batch_column = batch_column,
            sample_identifier_column = sample_identifier_column,
            transformer = self._transformer,
            truncate_max_range = truncate_max_range,
            output_directory = output_directory,
            prefix = prefix
        )

    def run_anndata_setup(self,
                          adata: AnnData,
                          layer: str = "compensated",
                          reference_column: str = "reference",
                          reference_value: str = "ref",
                          batch_column: str = "batch",
                          sample_identifier_column: str = "file_name",
                          channels: Union[list[str], str, Literal["all", "markers"]] = "markers",  # noqa
                          key_added: str = "cyto_normalized",
                          copy: bool = False
                          ) -> None:
        """\
        Method to setup the data handling for anndata objects. Will
        instantiate a :class:`~cytonormpy.DataHandlerAnnData` object.

        Parameters
        ----------
        adata
            The AnnData object
        layer
            The layer in `adata.uns` containing the compensated
            expression values
        reference_column
            The column in `adata.obs` that specifies whether a sample
            is used for reference and is therefore present in all batches.
        reference_value
            Specifies the value that is considered a reference. Defaults to
            'ref'
        batch_column
            The column in `adata.obs` that specifies the batch.
        sample_identifier_column
            Specifies the column in `adata.obs` that is unique to the samples.
        channels
            Can be a list of detectors (e.g. BV421-A), a single
            channel or 'all' or 'markers'. If `markers`, channels
            containing 'FSC', 'SSC', 'Time', 'AF' and CyTOF technicals
            will be excluded.
        key_added
            The name of the layer in `adata.layers` where the
            normalized data are inserted to.

        Returns
        -------
        None, appends `._datahandler` attribute.

        """
        adata = adata.copy() if copy else adata
        self._datahandler: DataHandler = DataHandlerAnnData(
            adata = adata,
            layer = layer,
            reference_column = reference_column,
            reference_value = reference_value,
            batch_column = batch_column,
            sample_identifier_column = sample_identifier_column,
            channels = channels,
            key_added = key_added,
            transformer = self._transformer
        )

    def add_transformer(self,
                        transformer: Transformer) -> None:
        """\
        Adds a transformer to transform the data to the `log`,
        `logicle`, `hyperlog` or `asinh` space.

        Parameters
        ----------
        transformer
            Transformer of class :class:`~cytonormpy.Transformer`

        Returns
        -------
        None

        """
        self._transformer = transformer

    def add_clusterer(self,
                      clusterer: ClusterBase) -> None:
        """\
        Adds a clusterer instance to transform the data to the `log`,
        `logicle`, `hyperlog` or `asinh` space.

        Parameters
        ----------
        clusterer
            Clusterer of class :class:`~cytonormpy.Clusterer`

        Returns
        -------
        None

        """
        self._clustering: ClusterBase = clusterer

    def run_clustering(self,
                       n_cells: Optional[int] = None,
                       test_cluster_cv: bool = True,
                       cluster_cv_threshold = 2,
                       **kwargs
                       ) -> None:
        """\
        Runs the clustering step. The clustering will be performed
        on as many cells as n_cells specifies. The remaining cells
        are then imputed from the initial clustering information.
        For FlowSOM, unclassified cells will be subjected to the same
        SOM and predicted.

        Parameters
        ----------
        n_cells
            Number of cells used for training. If None, all cells
            are used for training. Defaults to None.
        test_cluster_cv
            If True, CV of clusters per batch are calculated. Raises
            a warning if the calculated CV is above the threshold.
        cluster_cv_threshold
            The CV cutoff that is used to determine the appropriateness
            of the clustering.
        kwargs
            keyword arguments ultimately passed to the `train` function
            of the clusterer. Refer to the respective documentation.

        Returns
        -------
        None

        """
        if n_cells is not None:
            train_data_df = self._datahandler.get_ref_data_df_subsampled(
                n = n_cells
            )
        else:
            train_data_df = self._datahandler.get_ref_data_df()

        # we switch to numpy
        train_data = train_data_df.to_numpy(copy = True)

        self._clustering.train(X = train_data,
                               **kwargs)

        ref_data_df = self._datahandler.get_ref_data_df()

        # we switch to numpy
        ref_data_array = ref_data_df.to_numpy(copy = True)

        ref_data_df["clusters"] = self._clustering.calculate_clusters(X = ref_data_array)
        ref_data_df = ref_data_df.set_index("clusters", append = True)

        # we give it back to the data handler
        self._datahandler.ref_data_df = ref_data_df

        if test_cluster_cv:
            appropriate = _all_cvs_below_cutoff(
                df = self._datahandler.get_ref_data_df(),
                sample_key = self._datahandler._sample_identifier_column,
                cluster_key = "clusters",
                cv_cutoff = cluster_cv_threshold
            )
            if not appropriate:
                msg = "Cluster CV were above the threshold. "
                msg += "Calculating the quantiles on clusters "
                msg += "may not be appropriate. "
                warnings.warn(
                    msg,
                    ClusterCVWarning
                )

    def calculate_quantiles(self,
                            n_quantiles: int = 99,
                            min_cells: int = 50,
                            quantile_array: Optional[Union[list[float], np.ndarray]] = None
                            ) -> None:
        """\
        Calculates quantiles per batch, cluster and sample.

        Parameters
        ----------
        n_quantiles
            Number of quantiles to be calculated. Defaults to 99,
            which gives every percentile from 0.01 to 0.99.
        min_cells
            Minimum cells per cluster in order to calculate quantiles.
            If the cluster falls short, no quantiles and therefore
            no spline function is calculated. In that case, the spline
            function will return the input values. Defaults to 50.
        quantile_array
            Contains user-defined quantiles passed by the user. `n_quantiles`
            will be ignored. Has to contain numbers between 0 and 1.

        Returns
        -------
        None. Quantiles will be saved and used for later analysis.

        """

        if quantile_array is not None:
            if not isinstance(quantile_array, np.ndarray) and not isinstance(quantile_array, list):
                raise TypeError("quantile_array has to be passed as a list or an array")
            if np.max(quantile_array) > 1 or np.min(quantile_array) < 0:
                raise ValueError("Quantiles have to be between 0 and 1")

        ref_data_df: pd.DataFrame = self._datahandler.get_ref_data_df()

        if "clusters" not in ref_data_df.index.names:
            warnings.warn("No Clusters have been found.", UserWarning)
            ref_data_df["clusters"] = -1
            ref_data_df.set_index("clusters", append = True, inplace = True)

        batches = sorted(
            ref_data_df.index \
            .get_level_values("batch") \
            .unique() \
            .tolist()
        )
        clusters = sorted(
            ref_data_df.index \
            .get_level_values("clusters") \
            .unique() \
            .tolist()
        )
        channels = ref_data_df.columns.tolist()

        self.batches = batches
        self.clusters = clusters
        self.channels = channels

        n_channels = len(ref_data_df.columns)
        n_batches = len(batches)
        n_clusters = len(clusters)

        self._expr_quantiles = ExpressionQuantiles(
            n_channels = n_channels,
            n_quantiles = n_quantiles,
            n_batches = n_batches,
            n_clusters = n_clusters,
            quantile_array = quantile_array
        )

        # we store the clusters that could not be calculated for later.
        self._not_calculated = {
            batch: [] for batch in self.batches
        }

        ref_data_df = ref_data_df.sort_index(
            level = ["batch", "clusters"]
        )

        # we extract the values for batch and cluster...
        batch_idxs = ref_data_df.index.get_level_values("batch").to_numpy()
        cluster_idxs = ref_data_df.index.get_level_values("clusters").to_numpy()

        # ... and get the idxs of their unique combinations
        batch_cluster_idxs = np.vstack([batch_idxs, cluster_idxs]).T
        unique_combinations, batch_cluster_unique_idxs = np.unique(
            batch_cluster_idxs,
            axis = 0,
            return_index = True
        )
        # we append the shape as last idx
        batch_cluster_unique_idxs = np.hstack(
            [
                batch_cluster_unique_idxs,
                np.array(
                    batch_cluster_idxs.shape[0]
                )
            ]
        )

        # we create a lookup table to get the batch and cluster back
        batch_cluster_lookup = {
            idx: unique_combinations[i]
            for i, idx in enumerate(batch_cluster_unique_idxs[:-1])
        }
        # we also create a lookup table for the batch indexing...
        self.batch_idx_lookup = {
            batch: i
            for i, batch in enumerate(batches)
        }
        # ... and the cluster indexing
        cluster_idx_lookup = {
            cluster: i
            for i, cluster in enumerate(clusters)
        }
        
        # finally, we convert to numpy
        # As the array is sorted, we can index en bloc
        # with a massive speed improvement compared to
        # the pd.loc[] functionality.
        ref_data = ref_data_df.to_numpy()

        for i in range(batch_cluster_unique_idxs.shape[0]-1):
            batch, cluster = batch_cluster_lookup[batch_cluster_unique_idxs[i]]
            b = self.batch_idx_lookup[batch]
            c = cluster_idx_lookup[cluster]
            data = ref_data[
                batch_cluster_unique_idxs[i] : batch_cluster_unique_idxs[i+1],
                :
            ]
            if data.shape[0] < min_cells:
                warning_msg = f"{data.shape[0]} cells detected in batch "
                warning_msg += f"{batch} for cluster {cluster}. "
                warning_msg += "Skipping quantile calculation. "

                warnings.warn(
                    warning_msg,
                    UserWarning
                )
                self._not_calculated[batch].append(cluster)

                self._expr_quantiles.add_nan_slice(
                    batch_idx = b,
                    cluster_idx = c
                )

                continue

            self._expr_quantiles.calculate_and_add_quantiles(
                data = data,
                batch_idx = b,
                cluster_idx = c
            )
        return

    def calculate_splines(self,
                          limits: Optional[Union[list[float], np.ndarray]] = None,
                          goal: Union[str, int] = "batch_mean"
                          ) -> None:
        """\
        Calculates the spline functions of the expression values
        and the goal expression. The goal expression is calculated
        via the function specified in `goal`.

        Parameters
        ----------
        limits
            A list or array of fixed intensity values. These values will be
            appended to the calculated quantiles and included to calculate
            the spline functions. By default, the spline functions are extrapolated
            linearly outside the observed data range, using the slope at the last
            data point as the slope for the extrapolation. Use `limits` to further
            control the behaviour outside the data range.
        goal
            Function to calculate the goal expression per cluster and
            channel. Defaults to `batch_mean`, which calculates the mean
            over all batches. Can be any of `batch_median`, `batch_mean`
            or a specific batch passed as an int.

        Returns
        -------
        None


        """
        if limits is not None:
            if not isinstance(limits, list) and not isinstance(limits, np.ndarray):
                raise TypeError("Limits have to be passed as a list or array")

        expr_quantiles = self._expr_quantiles

        if isinstance(goal, int):
            goal = self.batch_idx_lookup[goal]

        # we now create the goal distributions with shape
        # n_channels x n_quantles x n_metaclusters x 1
        self._goal_distrib = GoalDistribution(expr_quantiles, goal = goal)
        goal_distrib = self._goal_distrib

        # Next, splines are calculated per channel, cluster and batch.
        # We store it in a Splines object, a fancy wrapper for a dictionary
        # of shape {batch: {cluster: {channel: splinefunc, ...}}}
        splines = Splines(batches = self.batches,
                          clusters = self.clusters,
                          channels = self.channels)

        for b, batch in enumerate(self.batches):
            for c, cluster in enumerate(self.clusters):
                if cluster in self._not_calculated[batch]:
                    for channel in self.channels:
                        spl = Spline(batch,
                                     cluster,
                                     channel,
                                     spline_calc_function = IdentitySpline,
                                     limits = limits)
                        spl.fit(current_distribution = None,
                                goal_distribution = None)
                        splines.add_spline(spl)
                else:
                    for ch, channel in enumerate(self.channels):
                        q = expr_quantiles.get_quantiles(channel_idx = ch,
                                                         quantile_idx = None,
                                                         cluster_idx = c,
                                                         batch_idx = b)
                        g = goal_distrib.get_quantiles(channel_idx = ch,
                                                       quantile_idx = None,
                                                       cluster_idx = c,
                                                       batch_idx = None)
                        spl = Spline(batch = batch,
                                     cluster = cluster,
                                     channel = channel,
                                     limits = limits)
                        spl.fit(q, g)
                        splines.add_spline(spl)

        self.splinefuncs = splines

        return

    def _normalize_file(self,
                        df: pd.DataFrame,
                        batch: str) -> pd.DataFrame:
        """\
        Private function to run the normalization. Can be
        called from self.normalize_data() and self.normalize_file().
        """

        data = df.to_numpy(copy = True)
        
        if self._clustering is not None:
            df["clusters"] = self._clustering.calculate_clusters(data)
        else:
            df["clusters"] = -1
        df = df.set_index("clusters", append = True)
        df["original_idx"] = list(range(df.shape[0]))
        df = df.set_index("original_idx", append = True)
        df = df.sort_index(level = "clusters")

        expr_data = df.to_numpy(copy = True)
        clusters, cluster_idxs = np.unique(
            df.index.get_level_values("clusters").to_numpy(),
            return_index = True
        )
        cluster_idxs = np.append(cluster_idxs, df.shape[0])
        channel_names = df.columns.tolist()

        for i, cluster in enumerate(clusters):
            row_slice = slice(cluster_idxs[i], cluster_idxs[i + 1])
            expr_data_to_pass = expr_data[
                row_slice,
                :
            ]
            assert expr_data_to_pass.shape[1] == len(self._datahandler._channel_indices)
            expr_data[
                row_slice,
                :
            ] = self._run_spline_funcs(
                    data = expr_data_to_pass,
                    channel_names = channel_names,
                    batch = batch,
                    cluster = cluster,
            )
        res = pd.DataFrame(
            data = expr_data,
            columns = df.columns,
            index = df.index
        )

        return res.sort_index(level = "original_idx", ascending = True)

    def _run_normalization(self,
                           file: str) -> None:
        """\
        wrapper function to coordinate the normalization and file writing
        in order to allow for parallelisation.
        """
        df = self._datahandler.get_dataframe(file_name = file)

        batch = self._datahandler.get_batch(file_name = file)

        df = self._normalize_file(df = df,
                                  batch = batch)

        self._datahandler.write(file_name = file,
                                data = df)

        print(f"normalized file {file}")

        return

    def normalize_data(self,
                       adata: Optional[AnnData] = None,
                       file_names: Optional[Union[list[str], str]] = None,
                       batches: Optional[Union[list[Union[str, int]], Union[str, int]]] = None,
                       n_jobs: int = 8) -> None:
        """\
        Applies the normalization procedure to the files and writes
        the data to disk or to the anndata file.

        Use the `file_names` and `batches` parameters to normalize data
        that were not yet available upon setup.

        Parameters
        ----------
        adata
            Optional. If passed, it is assumed that the original anndata
            object has been modified. If left `None`, the original
            anndata will be used.
        file_names:
            Optional. If left `None`, the validation files from the
            cytonorm object will be used.
        batches
            Optional. Specifies the batches of `file_names`.
        n_jobs
            Number of threads used for data analysis.
        
        Returns
        -------
        None

        """
        if adata is not None:
            assert isinstance(self._datahandler, DataHandlerAnnData)
            self._datahandler.adata = adata
            self._datahandler._provider._adata = adata

        if file_names is None:
            file_names = self._datahandler.validation_file_names
        else:
            assert batches is not None
            if not isinstance(file_names, list):
                file_names = [file_names]
            if not isinstance(batches, list):
                batches = [batches]
            if not len(file_names) == len(batches):
                raise ValueError("Please provide a batch for every file.")
            for file_name, batch in zip(file_names, batches):
                self._datahandler._add_file(file_name, batch)

        with cf.ThreadPoolExecutor(max_workers = n_jobs) as p:
            p.map(self._run_normalization, [file for file in file_names])

    def _run_spline_funcs(self,
                          data: np.ndarray,
                          channel_names: list[str],
                          batch: str,
                          cluster: str,
                          ) -> np.ndarray:
        """\
        Runs the spline function for the corresponding batch and cluster.
        Loops through all channels and repopulates the dataframe.
        """
        for ch_idx, channel in enumerate(channel_names):
            spline_func = self.splinefuncs.get_spline(
                batch = batch,
                cluster = cluster,
                channel = channel
            )
            vals = spline_func.transform(data[:, ch_idx])
            data[:, ch_idx] = vals

        return data


    def save_model(self,
                   filename: Union[PathLike, str] = "model.cytonorm") -> None:
        """\
        Function to save the current CytoNorm instance to disk.

        Parameters
        ----------
        filename:
            PathLike input to save the file. Defaults to 'model.cytonorm'.

        Returns
        -------
        None
        """
        import pickle
        with open(filename, "wb") as file:
            pickle.dump(self, file)


def read_model(filename: Union[PathLike, str]) -> CytoNorm:
    """\
    Read a model from disk.

    Parameters
    ----------
    filename:
        The filename of the model.

    Returns
    -------
    A :class:`~cytonormpy.CytoNorm` object.

    """
    with open(filename, "rb") as file:
        cytonorm_obj = pickle.load(file)
    return cytonorm_obj
