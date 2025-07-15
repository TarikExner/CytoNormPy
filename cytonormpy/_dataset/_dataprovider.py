import numpy as np
import pandas as pd

from abc import abstractmethod
from anndata import AnnData
from os import PathLike

from typing import Union, cast, Optional

from ._datareader import DataReaderFCS
from ._metadata import Metadata
from .._transformation._transformations import Transformer


class DataProvider:
    """\
    Base class for the data provider.
    """

    def __init__(self, metadata: Metadata, channels: Optional[list[str]], transformer):
        self.metadata = metadata
        self._channels = channels
        self._transformer = transformer

    @abstractmethod
    def parse_raw_data(self, file_name: str) -> pd.DataFrame:
        pass

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channels: list[str]):
        self._channels = channels

    def select_channels(self, data: pd.DataFrame) -> pd.DataFrame:
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
            return cast(pd.DataFrame, data[self._channels])
        return data

    @property
    def transformer(self):
        return self._transformer

    @transformer.setter
    def transformer(self, transformer: Transformer):
        self._transformer = transformer

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
                data=self._transformer.transform(data.values),
                columns=data.columns,
                index=data.index,
            )
        return data

    def inverse_transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
                data=self._transformer.inverse_transform(data.values),
                columns=data.columns,
                index=data.index,
            )
        return data

    def _annotate_sample_identifier(self, data: pd.DataFrame, file_name: str) -> pd.DataFrame:
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
        data[self.metadata.sample_identifier_column] = file_name
        return data

    def _annotate_reference_value(self, data: pd.DataFrame, file_name: str) -> pd.DataFrame:
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
        ref_value = self.metadata.get_ref_value(file_name)
        data[self.metadata.reference_column] = ref_value
        return data

    def _annotate_batch_value(self, data: pd.DataFrame, file_name: str) -> pd.DataFrame:
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
        batch_value = self.metadata.get_batch(file_name)
        data[self.metadata.batch_column] = batch_value
        return data

    def annotate_metadata(self, data: pd.DataFrame, file_name: str) -> pd.DataFrame:
        ref_value = self.metadata.get_ref_value(file_name)
        batch_value = self.metadata.get_batch(file_name)
        sample_identifier = file_name
        
        data.index = pd.MultiIndex.from_tuples(
            [(ref_value,batch_value,sample_identifier)] * len(data),
            names=[
                self.metadata.reference_column,
                self.metadata.batch_column,
                self.metadata.sample_identifier_column
            ]
        )
        return data

    def annotate_metadata_old(self, data: pd.DataFrame, file_name: str) -> pd.DataFrame:
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
                self.metadata.reference_column,
                self.metadata.batch_column,
                self.metadata.sample_identifier_column,
            ]
        )
        return data

    def prep_dataframe(self, file_name: str) -> pd.DataFrame:
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
        data = self.parse_raw_data(file_name)
        data = self.annotate_metadata(data, file_name)
        data = self.select_channels(data)
        data = self.transform_data(data)
        return data

    def subsample_df(self, df: pd.DataFrame, n: int):
        return df.sample(n=n, axis=0, random_state=187)


class DataProviderFCS(DataProvider):
    """\
    Class to handle the data providing for FCS files.
    This class will prepare a dataframe where the data
    are annotated with the metadata and the relevant
    channel data will be transformed.
    """

    def __init__(
        self,
        input_directory: Union[PathLike, str],
        metadata: Metadata,
        truncate_max_range: bool = False,
        channels: Optional[list[str]] = None,
        transformer: Optional[Transformer] = None,
    ) -> None:
        super().__init__(metadata=metadata, channels=channels, transformer=transformer)

        self._reader = DataReaderFCS(
            input_directory=input_directory, truncate_max_range=truncate_max_range
        )

    def parse_raw_data(self, file_name: str) -> pd.DataFrame:
        return self._reader.parse_fcs_df(file_name)


class DataProviderAnnData(DataProvider):
    """\
    Class to handle the data providing for anndata objects.
    This class will prepare a dataframe where the data
    are annotated with the metadata and the relevant
    channel data will be transformed.
    """

    def __init__(
        self,
        adata: AnnData,
        layer: str,
        metadata: Metadata,
        channels: Optional[list[str]] = None,
        transformer: Optional[Transformer] = None,
    ) -> None:
        super().__init__(metadata=metadata, channels=channels, transformer=transformer)

        self.adata = adata
        self.layer = layer

    def parse_raw_data(
        self, file_name: Union[str, list[str]], sample_identifier_column: Optional[str] = None
    ) -> pd.DataFrame:
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
        if not isinstance(file_name, list):
            files = [file_name]
        else:
            files = file_name
        return cast(
            pd.DataFrame,
            self.adata[self.adata.obs[self.metadata.sample_identifier_column].isin(files), :].to_df(
                layer=self.layer
            ),
        )
