import os
import pandas as pd
import numpy as np
import warnings

from os import PathLike
from anndata import AnnData
from flowio import FlowData
from flowio.exceptions import FCSParsingError
from pandas.io.parsers.readers import TextFileReader
from pandas.api.types import is_numeric_dtype

from typing import Union, Optional, Literal

from .._utils._utils import (_all_batches_have_reference,
                             _conclusive_reference_values)

from ._dataprovider import (DataProviderFCS,
                            DataProviderAnnData,
                            DataProvider)
from .._transformation._transformations import Transformer

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


    def __init__(self,
                 channels: Union[list[str], str, Literal["all", "markers"]],
                 provider: DataProvider):

        try:
            self._validation_value = list(set([
                val for val in self._metadata[self._reference_column]
                if val != self._reference_value
            ]))[0]
        except IndexError: # means we only have reference values
            self._validation_value = None

        self.ref_file_names = self._get_reference_file_names()
        self.validation_file_names = self._get_validation_file_names()
        self.all_file_names = self.ref_file_names + self.validation_file_names

        self._provider = provider

        self.ref_data_df = self._create_ref_data_df()

        self._all_detectors = self.ref_data_df.columns.tolist()
        _channel_user_input = channels
        self.channels: list[str] = self._select_channels(_channel_user_input)

        self._channel_indices = self._find_channel_indices()

    def _validate_metadata(self,
                           metadata: pd.DataFrame) -> None:
        self._metadata = metadata
        self._validate_metadata_table(self._metadata)
        self._validate_batch_references(self._metadata)
        self._convert_batch_dtype()

    def _convert_batch_dtype(self) -> None:
        """
        If the batch is entered as a string, we convert them
        to integers in order to comply with the numpy sorts
        later on.
        """
        if not is_numeric_dtype(self._metadata[self._batch_column]):
            try:
                self._metadata[self._batch_column] = \
                    self._metadata[self._batch_column].astype(np.int8)
            except ValueError:
                self._metadata[f"original_{self._batch_column}"] = \
                    self._metadata[self._batch_column]
                mapping = {entry: i for i, entry in enumerate(self._metadata[self._batch_column].unique())}
                self._metadata[self._batch_column] = \
                    self._metadata[self._batch_column].map(mapping)

    @abstractmethod
    def write(self,
              file_name: str,
              data: pd.DataFrame) -> None:
        pass

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

    def _add_file_to_metadata(self,
                              file_name,
                              batch):
        new_file_df = pd.DataFrame(
            data = [[file_name, self._validation_value, batch]],
            columns = [
                self._sample_identifier_column,
                self._reference_column,
                self._batch_column
            ],
            index = [-1]
        )
        self._metadata = pd.concat([self._metadata, new_file_df], axis = 0).reset_index(drop = True)
        self._provider._metadata = self._metadata

    def _add_file(self,
                  file_name,
                  batch):
        self._add_file_to_metadata(file_name, batch)
        if isinstance(self, DataHandlerAnnData):
            obs_idxs = self._find_obs_idxs(file_name)
            arr_idxs = self._get_array_indices(obs_idxs)
            self._copy_input_values_to_key_added(arr_idxs)

    def _init_metadata_columns(self,
                               reference_column: str,
                               reference_value: str,
                               batch_column: str,
                               sample_identifier_column) -> None:
        self._reference_column = reference_column
        self._reference_value = reference_value
        self._batch_column = batch_column
        self._sample_identifier_column = sample_identifier_column

        return

    def get_batch(self,
                  file_name: str) -> str:
        """\
        Returns the corresponding batch of a file.

        Parameters
        ----------
        file_name
            The sample identifier.

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

    def get_dataframe(self,
                      file_name: str) -> pd.DataFrame:
        """
        Returns a dataframe for the indicated file name.

        Parameters
        ----------
        file_name
            The file_name of the file being read.

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.
        """

        return self._provider.prep_dataframe(file_name)

    def get_corresponding_ref_dataframe(self,
                                        file_name: str) -> pd.DataFrame:
        """
        Returns the data of the corresponding reference
        for the indicated file name.

        Parameters
        ----------
        file_name
            The file_name of the file being read.

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.
        """
        corresponding_reference_file = \
            self._find_corresponding_reference_file(file_name)
        return self.get_dataframe(file_name = corresponding_reference_file)


    def _create_ref_data_df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self._provider.prep_dataframe(file)
                for file in self.ref_file_names
            ],
            axis = 0
        )

    def get_ref_data_df_subsampled(self,
                                   n: int):
        """
        Returns the reference data frame, subsampled to
        `n` events.

        Parameters
        ----------
        n
            The number of events to be subsampled.

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.
        """
        assert isinstance(self.ref_data_df, pd.DataFrame)
        return self._subsample_df(self.ref_data_df, n)

    def _subsample_df(self,
                      df: pd.DataFrame,
                      n: int):
        return df.sample(n = n, axis = 0, random_state = 187)

    def get_ref_data_df(self) -> pd.DataFrame:
        """
        Returns the reference data frame.

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.
        """
        assert isinstance(self.ref_data_df, pd.DataFrame)
        return self.ref_data_df

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
        detectors = self._all_detectors
        return np.array(
            [
                detectors.index(ch) for ch in detectors
                if ch in self.channels
            ]
        )

    def _find_channel_indices_in_fcs(self,
                                     pnn_labels: dict[str, int],
                                     cytonorm_channels: pd.Index):
        return [
            pnn_labels[channel] - 1
            for channel in cytonorm_channels
        ]

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
    channels
        Can be a list of detectors (e.g. BV421-A), a single
        channel or 'all' or 'markers'. If `markers`, channels
        containing 'FSC', 'SSC', 'Time', 'AF' and CyTOF technicals
        will be excluded.
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
                 channels: Union[list[str], str, Literal["all", "markers"]] = "markers",  # noqa
                 reference_column: str = "reference",
                 reference_value: str = "ref",
                 batch_column: str = "batch",
                 sample_identifier_column: str = "file_name",
                 transformer: Optional[Transformer] = None,
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
            _metadata = metadata
        else:
            _metadata = self._read_metadata(metadata)

        self._validate_metadata(_metadata)


        _provider = self._create_data_provider(
            input_directory = self._input_dir,
            truncate_max_range = truncate_max_range,
            sample_identifier_column = sample_identifier_column,
            reference_column = reference_column,
            batch_column = batch_column,
            metadata = _metadata,
            channels = None, # instantiate with None as we dont know the channels yet
            transformer = transformer
        )


        super().__init__(
            channels = channels,
            provider = _provider,
        )

        self._provider.channels = self.channels
        self.ref_data_df = self._provider.select_channels(self.ref_data_df)

    def _create_data_provider(self,
                              input_directory,
                              metadata: pd.DataFrame,
                              channels: Optional[list[str]],
                              reference_column: str = "reference",
                              batch_column: str = "batch",
                              sample_identifier_column: str = "file_name",
                              truncate_max_range: bool = True,
                              transformer: Optional[Transformer] = None) -> DataProviderFCS:
        return DataProviderFCS(
            input_directory = input_directory,
            truncate_max_range = truncate_max_range,
            sample_identifier_column = sample_identifier_column,
            reference_column = reference_column,
            batch_column = batch_column,
            metadata = metadata,
            channels = channels,
            transformer = transformer
        )

    def _read_metadata(self,
                       path: PathLike) -> pd.DataFrame:
        delimiter = self._fetch_delimiter(path)
        return pd.read_csv(path, sep = delimiter, index_col = False)

    def _fetch_delimiter(self,
                         path: PathLike) -> str:
        reader: TextFileReader = pd.read_csv(path,
                                             sep = None,
                                             iterator = True,
                                             engine = "python")
        return reader._engine.data.dialect.delimiter

    def write(self,
              file_name: str,
              data: pd.DataFrame,
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

        """function to load the fcs from the hard drive"""
        try:
            ignore_offset_error = False
            fcs = FlowData(
                file_path,
                ignore_offset_error
            )
        except FCSParsingError:
            ignore_offset_error = False
            warnings.warn(
                "CytoNormPy IO: FCS file could not be read with "
                f"ignore_offset_error set to {ignore_offset_error}. "
                "Parameter is set to True."
            )
            fcs = FlowData(
                file_path,
                ignore_offset_error = True
            )

        channels: dict = fcs.channels

        pnn_labels = {
            channels[channel_number]["PnN"]: int(channel_number)
            for channel_number in channels
        }

        channel_indices = self._find_channel_indices_in_fcs(pnn_labels,
                                                            data.columns)
        orig_events = np.reshape(
            np.array(fcs.events),
            (-1, fcs.channel_count)
        )
        inv_transformed: pd.DataFrame = self._provider.inverse_transform_data(data)
        orig_events[:, channel_indices] = inv_transformed.values
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
                 transformer: Optional[Transformer] = None,
                 key_added: str = "cyto_normalized"):
        self.adata = adata
        self._layer = layer
        self._key_added = key_added

        # We copy the input data to the newly created layer
        # to ensure that non-normalized data stay as the input
        if self._key_added not in self.adata.layers:
            self.adata.layers[self._key_added] = \
                np.array(self.adata.layers[self._layer])

        self._init_metadata_columns(
            reference_column = reference_column,
            reference_value = reference_value,
            batch_column = batch_column,
            sample_identifier_column = sample_identifier_column
        )

        _metadata = self._condense_metadata(
            self.adata.obs,
            reference_column,
            batch_column,
            sample_identifier_column
        )

        self._validate_metadata(_metadata)

        _provider = self._create_data_provider(
            adata = adata,
            layer = layer,
            sample_identifier_column = sample_identifier_column,
            reference_column = reference_column,
            batch_column = batch_column,
            metadata = _metadata,
            channels = None, # instantiate with None as we dont know the channels yet
            transformer = transformer
        )

        super().__init__(
            channels = channels,
            provider = _provider,
        )

        self._provider.channels = self.channels
        self.ref_data_df = self._provider.select_channels(self.ref_data_df)

        # TODO: add check for anndata obs

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

    def _create_data_provider(self,
                              adata: AnnData,
                              layer: str,
                              reference_column: str,
                              batch_column: str,
                              sample_identifier_column: str,
                              channels: Optional[list[str]],
                              metadata: pd.DataFrame,
                              transformer: Optional[Transformer] = None) -> DataProviderAnnData:
        return DataProviderAnnData(
            adata = adata,
            layer = layer,
            sample_identifier_column = sample_identifier_column,
            reference_column = reference_column,
            batch_column = batch_column,
            metadata = metadata,
            channels = channels, # instantiate with None as we dont know the channels yet
            transformer = transformer
        )

    def _find_obs_idxs(self,
                       file_name) -> pd.Index:
        return self.adata.obs.loc[
            self.adata.obs[self._sample_identifier_column] == file_name,
            :
        ].index

    def _get_array_indices(self,
                           obs_idxs: pd.Index) -> np.ndarray:
        return self.adata.obs.index.get_indexer(obs_idxs)

    def _copy_input_values_to_key_added(self,
                                        idxs: np.ndarray) -> None:
        self.adata.layers[self._key_added][idxs, :] = \
            self.adata.layers[self._layer][idxs, :]

    def write(self,
              file_name: str,
              data: pd.DataFrame) -> None:
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
        obs_idxs = self._find_obs_idxs(file_name)

        # leaving at pd.Index type is 2x faster
        arr_idxs = self._get_array_indices(obs_idxs)

        channel_indices = self._find_channel_indices_in_adata(data.columns)

        inv_transformed: pd.DataFrame = self._provider.inverse_transform_data(data)

        self.adata.layers[self._key_added][
            np.ix_(arr_idxs, np.array(channel_indices))
        ] = inv_transformed.values

        return

    def _find_channel_indices_in_adata(self,
                                       channels: pd.Index) -> list[int]:
        adata_channels = self.adata.var.index.tolist()
        return [
            adata_channels.index(channel)
            for channel in channels
        ]

