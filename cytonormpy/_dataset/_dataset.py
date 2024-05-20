import os
from os import PathLike
import pandas as pd
import numpy as np
from anndata import AnnData
from flowio import FlowData

from typing import Union, Optional, Literal

from .._utils._utils import (_all_batches_have_reference,
                             _conclusive_reference_values)

from ._fcs_file import FCSFile

from abc import abstractmethod


class DataHandler:
    """\
    Base Class for data handling.
    """

    _flow_technicals: list[str] = [
        "fsc", "ssc", "time"
    ]
    _spectral_flow_technicals: list[str] = [
        "fsc", "ssc", "time", "af"
    ]
    _cytof_technicals: list[str] = [
        "event_length", "width", "height", "center",
        "residual", "offset", "amplitude", "dna1", "dna2"
    ]
    _all_detectors = []

    ref_file_names = []
    validation_file_names = []

    _metadata = pd.DataFrame()

    _reference_column = ""
    _reference_value = ""
    _batch_column = ""
    _sample_identifier_column = ""

    def __init__(self,
                 channels: Union[list[str], str, Literal["all", "markers"]],
                 ref_data_df: pd.DataFrame):
        self._all_detectors = ref_data_df.columns.tolist()
        _channel_user_input = channels
        self.channels: list[str] = self._select_channels(_channel_user_input)

        self._channel_indices = self._find_channel_indices()

        self.ref_data_df = ref_data_df[self.channels]

        self._all_file_names = self.ref_file_names + self.validation_file_names

    @abstractmethod
    def _create_ref_data_df(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _get_reference_data_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dataframe(self,
                      file_name: str,
                      raw: bool = False,
                      annot_file_name: bool = False) -> pd.DataFrame:
        pass

    def _init_metadata_columns(self,
                               reference_column: str,
                               reference_value: str,
                               batch_column: str,
                               sample_identifier_column) -> None:
        self._reference_column = reference_column
        self._reference_value = reference_value
        self._batch_column = batch_column
        self._sample_identifier_column = sample_identifier_column

    def _append_metadata_to_df(self,
                               df: pd.DataFrame,
                               file_name: str) -> pd.DataFrame:
        ref_value = self._metadata.loc[
            self._metadata[self._sample_identifier_column] == file_name,
            self._reference_column
        ].iloc[0]

        batch_value = self._metadata.loc[
            self._metadata[self._sample_identifier_column] == file_name,
            self._batch_column
        ].iloc[0]

        df[self._reference_column] = ref_value
        df[self._batch_column] = batch_value
        df[self._sample_identifier_column] = file_name

        return df

    def get_batch(self,
                  file_name: str) -> str:
        """\

        Returns
        -------
        The batch of the file specified in file_name.
        """

        return self._metadata.loc[
            self._metadata[self._sample_identifier_column] == file_name,
            self._batch_column
        ].iloc[0]

    def _find_corresponding_reference_file(self,
                                           file_name):
        batch = self.get_batch(file_name)
        return self._metadata.loc[
            (self._metadata[self._batch_column] == batch) &
            (self._metadata[self._reference_column] == self._reference_value),
            self._sample_identifier_column
        ].iloc[0]

    def get_corresponding_ref_dataframe(self,
                                        file_name: str,
                                        raw: bool = False,
                                        annot_file_name: bool = False
                                        ) -> pd.DataFrame:
        corresponding_reference_file = \
            self._find_corresponding_reference_file(file_name)
        return self.get_dataframe(file_name = corresponding_reference_file,
                                  raw = raw,
                                  annot_file_name = annot_file_name)

    @abstractmethod
    def write(self,
              file_name: str,
              data: np.ndarray) -> None:
        pass

    def get_ref_data_df(self) -> pd.DataFrame:
        assert isinstance(self.ref_data_df, pd.DataFrame)
        return self.ref_data_df

    def get_ref_data_df_subsampled(self,
                                   n: int):
        assert isinstance(self.ref_data_df, pd.DataFrame)
        return self._subsample_df(self.ref_data_df, n)

    def _subsample_df(self,
                      df: pd.DataFrame,
                      n: int):
        return df.sample(n = n, axis = 0, random_state = 187)

    def _select_channels(self,
                         user_input: Union[list[str], str, Literal["all", "markers"]]  # noqa
                         ) -> list[str]:
        """\
        function looks through the channels and decides which channels to keep
        based on the user input.
        """
        if user_input == "all":
            return self._all_detectors
        elif user_input == "markers":
            return self._find_marker_channels(self._all_detectors)
        else:
            assert isinstance(user_input, list), type(user_input)
            return [ch for ch in user_input if ch in self._all_detectors]

    def _find_marker_channels(self,
                              detectors: list[str]) -> list[str]:
        exclude = \
            self._flow_technicals + \
            self._cytof_technicals + \
            self._spectral_flow_technicals
        return [ch for ch in detectors if ch.lower() not in exclude]

    def _find_channel_indices(self) -> np.ndarray:
        return np.array(
            [i for i, ch in enumerate(self._all_detectors)
             if ch in self.channels]
        )

    def _get_reference_file_names(self) -> list[str]:
        return self._metadata.loc[
            self._metadata[self._reference_column] == self._reference_value,
            self._sample_identifier_column
        ].unique().tolist()

    def _get_validation_file_names(self) -> list[str]:
        return self._metadata.loc[
            self._metadata[self._reference_column] != self._reference_value,
            self._sample_identifier_column
        ].unique().tolist()

    def _validate_metadata_table(self,
                                 metadata: pd.DataFrame):
        if not all(k in metadata.columns
                   for k in [self._sample_identifier_column,
                             self._reference_column,
                             self._batch_column]):
            raise ValueError(
                "Metadata must contain the columns "
                f"[{self._sample_identifier_column}, "
                f"{self._reference_column}, "
                f"{self._batch_column}]. "
                f"Found {metadata.columns}"
            )
        if not _conclusive_reference_values(metadata,
                                            self._reference_column):
            raise ValueError(
                f"The column {self._reference_column} must only contain "
                "descriptive values for references and other values" 
            )

    def _validate_batch_references(self,
                                   metadata: pd.DataFrame):
        if not _all_batches_have_reference(
                metadata,
                reference = self._reference_column,
                batch = self._batch_column,
                ref_control_value = self._reference_value
        ):
            raise ValueError(
                "All batches must have reference samples."
            )

    def _subsample_array(self,
                         data: np.ndarray,
                         n: int) -> np.ndarray:
        idxs = np.random.choice(data.shape[0], n, replace = False)
        return data[idxs]

    def _subsample_reference_data(self,
                                  n_cells: Optional[int]) -> np.ndarray:

        ref_data_array = self._get_reference_data_array()

        if n_cells is None:
            return ref_data_array

        return self._subsample_array(ref_data_array, n_cells)

    @property
    def flow_technicals(self):
        return self._flow_technicals

    @flow_technicals.setter
    def flow_technicals(self,
                        technicals: list[str]):
        self._flow_technicals = technicals

    def append_flow_technicals(self,
                               value):
        self.flow_technicals.append(value)

    @property
    def spectral_flow_technicals(self):
        return self._spectral_flow_technicals

    @spectral_flow_technicals.setter
    def spectral_flow_technicals(self,
                                 technicals: list[str]):
        self._spectral_flow_technicals = technicals

    def append_spectral_flow_technicals(self,
                                        value):
        self.spectral_flow_technicals.append(value)

    @property
    def cytof_technicals(self):
        return self._cytof_technicals

    @cytof_technicals.setter
    def cytof_technicals(self,
                         technicals: list[str]):
        self._cytof_technicals = technicals

    def append_cytof_technicals(self,
                                value):
        self.cytof_technicals.append(value)


class DataHandlerFCS(DataHandler):
    """\
    Class to intermediately represent the data, read and
    write outputs and handle intermediate steps.

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
    None

    """

    def __init__(self,
                 metadata: Union[pd.DataFrame, PathLike],
                 input_directory: Optional[PathLike] = None,
                 reference_column: str = "reference",
                 reference_value: str = "ref",
                 batch_column: str = "batch",
                 sample_identifier_column: str = "file_name",
                 channels: Union[list[str], str, Literal["all", "markers"]] = "markers",  # noqa
                 truncate_max_range: bool = True,
                 output_directory: Optional[PathLike] = None,
                 prefix: str = "Norm"
                 ) -> None:
        self._input_dir = input_directory or os.getcwd()
        self._output_dir = output_directory or input_directory
        self._prefix = prefix

        self._init_metadata_columns(
            reference_column = reference_column,
            reference_value = reference_value,
            batch_column = batch_column,
            sample_identifier_column = sample_identifier_column
        )

        if isinstance(metadata, pd.DataFrame):
            self._metadata = metadata
        else:
            self._metadata = self._read_metadata(metadata)

        self._validate_metadata_table(self._metadata)
        self._validate_batch_references(self._metadata)

        self.ref_file_names = self._get_reference_file_names()
        self.validation_file_names = self._get_validation_file_names()

        self._truncate_max_range = truncate_max_range
        ref_data_df = self._create_ref_data_df()

        super().__init__(channels = channels,
                         ref_data_df = ref_data_df)

    def _create_ref_data_df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self._fcs_to_df(file)
                for file in
                self._get_reference_file_names()
            ]
        )

    def get_dataframe(self,
                      file_name: str,
                      raw: bool = False,
                      annot_file_name: bool = False) -> pd.DataFrame:
        """
        Returns a dataframe for the indicated file name.

        Parameters
        ----------
        file_name
            The file_name of the file being read.
        raw
            If True, returns full frame, otherwise subset
            for self.channels
        annot_file_name
            If True, appends a column with the file name
            using the sample identifier column used at
            setup.

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.
        """
        df = self._fcs_to_df(file_name)
        if raw is True:
            if annot_file_name:
                df[self._sample_identifier_column] = file_name
                return df
            return df
        df = df[self.channels]
        if annot_file_name:
            df[self._sample_identifier_column] = file_name
        assert isinstance(df, pd.DataFrame)
        return df

    def _fcs_to_df(self,
                   file_name) -> pd.DataFrame:
        """\
        Returns a DataFrame from an .fcs file.
        All channels are included and metadata ref and batch
        are appended and set as the index.
        """
        fcs = self._read_fcs_file(
            file_name = file_name,
        )
        df = fcs.to_df()
        df = self._append_metadata_to_df(df = df,
                                         file_name = file_name)

        assert all(k in df.columns for k in [self._reference_column,
                                             self._batch_column,
                                             self._sample_identifier_column])
        df = df.set_index(
            [
                self._reference_column,
                self._batch_column,
                self._sample_identifier_column
            ]
        )

        return df

    def _assert_panel_equal(self) -> bool:
        """
        checks panels of .fcs files to see if they are
        identical
        """
        return True

    def _read_fcs_file(self,
                       file_name: str) -> FCSFile:
        return FCSFile(
            input_directory = self._input_dir,
            file_name = file_name,
            truncate_max_range = self._truncate_max_range
        )

    def _read_metadata(self,
                       path: PathLike) -> pd.DataFrame:
        return pd.read_csv(path, index_col = False)

    def write(self,
              file_name: str,
              data: np.ndarray,
              output_dir: Optional[PathLike] = None) -> None:
        """\
        Writes the data to the hard drive as an .fcs file.

        Parameters
        ----------
        file_name
            The file name where the data are inserted to.
        data
            The data to be inserted.

        Returns
        -------
        None

        """
        file_path = os.path.join(self._input_dir, file_name)
        if output_dir is not None:
            new_file_path = os.path.join(
                output_dir, f"{self._prefix}_{file_name}"
            )
        else:
            assert self._output_dir is not None
            new_file_path = os.path.join(
                self._output_dir, f"{self._prefix}_{file_name}"
            )
        fcs = FlowData(file_path)
        orig_events = np.reshape(np.array(fcs.events),
                                 (-1, fcs.channel_count))
        orig_events[:, self._channel_indices] = data
        fcs.events = orig_events.flatten()  # type: ignore
        fcs.write_fcs(new_file_path, metadata = fcs.text)


class DataHandlerAnnData(DataHandler):
    """\
    Class to handle AnnData objects in cytonormpy.

    Parameters
    ----------
    adata
        The anndata object of shape n_objects x n_channels.
    layer
        The layer of the AnnData object to be used.
    reference_column
        The column in `adata.obs` that specifies whether files
        serve as a batch-reference
    reference_value
        The value of `reference_column` that specifies the
        reference files.
    batch_column
        The column in `adata.obs` that specifies which batch
        the files belong to.
    sample_identifier_column
        The column in `adata.obs` that specifies the individual
        files. Have to be unique.
    channels
        Can be a list of detectors (e.g. BV421-A), a single
        channel or `all` or `markers`. If `markers`, channels
        containing 'FSC', 'SSC', 'Time', 'AF' and CyTOF technicals
        will be excluded.
    key_added
        The name of the layer in `adata.layers` where the
        normalized data are inserted to.

    Returns
    -------
    None

    """

    def __init__(self,
                 adata: AnnData,
                 layer: str,
                 reference_column: str,
                 reference_value: str,
                 batch_column: str,
                 sample_identifier_column: str,
                 channels: Union[list[str], str, Literal["all", "marker"]],
                 key_added: str = "cyto_normalized"):
        self.adata = adata
        self._layer = layer
        self._key_added = key_added

        self._init_metadata_columns(
            reference_column = reference_column,
            reference_value = reference_value,
            batch_column = batch_column,
            sample_identifier_column = sample_identifier_column
        )

        # We copy the input data to the newly created layer
        # to ensure that non-normalized data stay as the input
        if self._key_added not in self.adata.layers:
            self.adata.layers[self._key_added] = \
                np.array(self.adata.layers[self._layer])

        self._metadata = self._condense_metadata(self.adata.obs,
                                                 reference_column,
                                                 batch_column,
                                                 sample_identifier_column)

        self._validate_metadata_table(self._metadata)
        self._validate_batch_references(self._metadata)

        self.ref_file_names = self._get_reference_file_names()
        self.validation_file_names = self._get_validation_file_names()

        ref_data_df = self._create_ref_data_df()
        super().__init__(channels = channels,
                         ref_data_df = ref_data_df)

        # TODO: add check for anndata obs

    def _create_ref_data_df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self._ad_to_df(file)
                for file in
                self._get_reference_file_names()
            ]
        )

    def _condense_metadata(self,
                           obs: pd.DataFrame,
                           reference_column: str,
                           batch_column: str,
                           sample_identifier_column: str) -> pd.DataFrame:
        df = obs[[reference_column,
                  batch_column,
                  sample_identifier_column]]
        df = df.drop_duplicates()
        assert isinstance(df, pd.DataFrame)
        return df

    def get_dataframe(self,
                      file_name: str,
                      raw: bool = False,
                      annot_file_name: bool = False) -> pd.DataFrame:
        """
        Returns a dataframe for the indicated file name.

        Parameters
        ----------
        file_name
            The file_name of the file being read.
        raw
            If True, returns full frame, otherwise subset
            for self.channels
        annot_file_name
            If True, appends a column with the file name
            using the sample identifier column used at
            setup.

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.

        """
        df = self._ad_to_df(file_name)
        if raw is True:
            if annot_file_name:
                df[self._sample_identifier_column] = file_name
                return df
            return df
        df = df[self.channels]
        if annot_file_name:
            df[self._sample_identifier_column] = file_name
        assert isinstance(df, pd.DataFrame)
        return df

    def _ad_to_df(self,
                  file_name: str):
        df = self.adata[
            self.adata.obs[self._sample_identifier_column] == file_name,
            :
        ].to_df(layer = self._layer)
        df = self._append_metadata_to_df(df = df,
                                         file_name = file_name)

        assert all(k in df.columns for k in [self._reference_column,
                                             self._batch_column])
        df = df.set_index(
            [
                self._reference_column,
                self._batch_column,
                self._sample_identifier_column
            ]
        )

        return df

    def write(self,
              file_name: str,
              data: np.ndarray) -> None:
        """\
        Writes the data to the anndata object to the layer
        specified during setup.

        Parameters
        ----------
        file_name
            The file name where the data are inserted to.
        data
            The data to be inserted.

        Returns
        -------
        None

        """

        obs_idxs = self.adata.obs.loc[
            self.adata.obs["file_name"] == file_name,
            :
        ].index

        # leaving at pd.Index type is 2x faster
        arr_idxs = self.adata.obs.index.get_indexer(obs_idxs)

        self.adata.layers[self._key_added][
            arr_idxs[:, np.newaxis],
            self._channel_indices[np.newaxis, :]
        ] = data

        return
