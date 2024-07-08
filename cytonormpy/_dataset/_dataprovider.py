import pandas as pd
from .._transformation._transformations import Transformer
from typing import Optional
from os import PathLike
from anndata import AnnData

from typing import Union

from ._datareader import DataReaderFCS

class DataProvider:
    """\
    Base class for the data provider.
    """

    def __init__(self,
                 sample_identifier_column,
                 reference_column,
                 batch_column,
                 metadata,
                 channels,
                 transformer):

        self._sample_identifier_column = sample_identifier_column
        self._reference_column = reference_column
        self._batch_column = batch_column
        self._metadata = metadata
        self._channels = channels
        self._transformer = transformer

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self,
                 channels: list[str]):
        self._channels = channels 

    def select_channels(self,
                        data: pd.DataFrame) -> pd.DataFrame:
        """\
        Subsets the channels in a dataframe.

        Parameters
        ----------
        data
            The expression data as a pandas DataFrame

        Returns
        -------
        The data subset for the channels stored in the `_channels`
        attribute.

        """
        if self._channels is not None:
            return data[self._channels]
        return data

    @property
    def transformer(self):
        return self._transformer

    @transformer.setter
    def transformer(self,
                    transformer: Transformer):
        self._transformer = transformer

    def transform_data(self,
                       data: pd.DataFrame) -> pd.DataFrame:
        """\
        Transforms the data according to the transformer added
        upon instantiation.

        Parameters
        ----------
        data
            The data passed as a pandas DataFrame.

        Returns
        -------
        Dependent on the transformer, the transformed or the raw data.

        """
        if self._transformer is not None:
            return pd.DataFrame(
                data = self._transformer.transform(data.values),
                columns = data.columns,
                index = data.index
            )
        return data

    def inverse_transform_data(self,
                               data: pd.DataFrame) -> pd.DataFrame:
        """\
        Inverse transforms the data according to the transformer added
        upon instantiation.

        Parameters
        ----------
        data
            The data passed as a pandas DataFrame.

        Returns
        -------
        Dependent on the transformer, the transformed or the raw data.

        """
        if self._transformer is not None:
            return pd.DataFrame(
                data = self._transformer.inverse_transform(data.values),
                columns = data.columns,
                index = data.index
            )
        return data

    def _annotate_sample_identifier(self,
                                    data: pd.DataFrame,
                                    file_name: str) -> pd.DataFrame:
        """\
        Annotates the sample identifier to the expression data.

        Parameters
        ----------
        data
            The data passed as a pandas DataFrame.
        file_name
            The file identifier that is used for the metadata lookup.

        Returns
        -------
        The annotated expression data.

        """
        data[self._sample_identifier_column] = file_name
        return data

    def _annotate_reference_value(self,
                                  data: pd.DataFrame,
                                  file_name: str) -> pd.DataFrame:
        """\
        Annotates the reference value to the expression data.

        Parameters
        ----------
        data
            The data passed as a pandas DataFrame.
        file_name
            The file identifier that is used for the metadata lookup.

        Returns
        -------
        The annotated expression data.

        """
        ref_value = self._metadata.loc[
            self._metadata[self._sample_identifier_column] == file_name,
            self._reference_column
        ].iloc[0]
        data[self._reference_column] = ref_value
        return data

    def _annotate_batch_value(self,
                              data: pd.DataFrame,
                              file_name: str) -> pd.DataFrame:
        """\
        Annotates the batch number to the expression data.

        Parameters
        ----------
        data
            The data passed as a pandas DataFrame.
        file_name
            The file identifier that is used for the metadata lookup.

        Returns
        -------
        The annotated expression data.

        """
        batch_value = self._metadata.loc[
            self._metadata[self._sample_identifier_column] == file_name,
            self._batch_column
        ].iloc[0]
        data[self._batch_column] = batch_value
        return data

    def annotate_metadata(self,
                          data: pd.DataFrame,
                          file_name: str) -> pd.DataFrame:
        """\
        Annotates metadata (sample identifier, batch value and
        reference value) to the expression data.

        Parameters
        ----------
        data
            The data passed as a pandas DataFrame.
        file_name
            The file identifier that is used for the metadata lookup.

        Returns
        -------
        The annotated expression data.

        """

        self._annotate_reference_value(data, file_name)
        self._annotate_batch_value(data, file_name)
        self._annotate_sample_identifier(data, file_name)
        data = data.set_index(
            [
                self._reference_column,
                self._batch_column,
                self._sample_identifier_column
            ]
        )

        return data


class DataProviderFCS(DataProvider):
    """\
    Class to handle the data providing for FCS files.
    This class will prepare a dataframe where the data
    are annotated with the metadata and the relevant
    channel data will be transformed.
    """

    def __init__(self,
                 input_directory: Union[PathLike, str],
                 truncate_max_range: bool = False,
                 sample_identifier_column: Optional[str] = None,
                 reference_column: Optional[str] = None,
                 batch_column: Optional[str] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 channels: Optional[list[str]] = None,
                 transformer: Optional[Transformer] = None) -> None:

        super().__init__(
            sample_identifier_column = sample_identifier_column,
            reference_column = reference_column,
            batch_column = batch_column,
            metadata = metadata,
            channels = channels,
            transformer = transformer
        )

        self._reader = DataReaderFCS(
            input_directory = input_directory,
            truncate_max_range = truncate_max_range
        )

    def prep_dataframe(self,
                       file_name: str) -> pd.DataFrame:
        """\
        Prepares the dataframe by annotating metadata,
        selecting the relevant channels and transforming.
        
        Parameters
        ----------
        file_name
            The file identifier of which the data are provided

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.

        """
        data = self._reader.parse_fcs_df(file_name)
        data = self.annotate_metadata(data, file_name)
        data = self.select_channels(data)
        data = self.transform_data(data)
        return data


class DataProviderAnnData(DataProvider):
    """\
    Class to handle the data providing for anndata objects.
    This class will prepare a dataframe where the data
    are annotated with the metadata and the relevant
    channel data will be transformed.
    """

    def __init__(self,
                 adata: AnnData,
                 layer: str,
                 sample_identifier_column: Optional[str] = None,
                 reference_column: Optional[str] = None,
                 batch_column: Optional[str] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 channels: Optional[list[str]] = None,
                 transformer: Optional[Transformer] = None) -> None:

        super().__init__(
            sample_identifier_column = sample_identifier_column,
            reference_column = reference_column,
            batch_column = batch_column,
            metadata = metadata,
            channels = channels,
            transformer = transformer
        )

        self._adata = adata
        self._layer = layer

    def parse_anndata_df(self,
                         file_names: Union[list[str], str]) -> pd.DataFrame:
        """\
        Parses the expression data stored in the anndata object by the
        sample identifier.
        
        Parameters
        ----------
        file_name
            The file identifier of which the data are provided. Can be
            a list of files.

        Returns
        -------
        A :class:`pandas.DataFrame` containing the raw expression data
        of the specified file.

        """
        if not isinstance(file_names, list):
            file_names = [file_names]
        return self._adata[
            self._adata.obs[self._sample_identifier_column].isin(file_names),
            :
        ].to_df(layer = self._layer)

    def prep_dataframe(self,
                       file_name: str) -> pd.DataFrame:
        """\
        Prepares the dataframe by annotating metadata,
        selecting the relevant channels and transforming.
        
        Parameters
        ----------
        file_name
            The file identifier of which the data are provided

        Returns
        -------
        A :class:`pandas.DataFrame` containing the expression data.

        """
        data = self.parse_anndata_df(file_name)
        data = self.annotate_metadata(data, file_name)
        data = self.select_channels(data)
        data = self.transform_data(data)
        return data

