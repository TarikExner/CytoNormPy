import pandas as pd
from .._transformation._transformations import Transformer
from typing import Optional
from os import PathLike
from anndata import AnnData

from ._datareader import DataReaderFCS

class DataProvider:

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
        if self._transformer is not None:
            return pd.DataFrame(
                data = self._transformer.transform(data.values),
                columns = data.columns,
                index = data.index
            )
        return data

    def inverse_transform_data(self,
                               data: pd.DataFrame) -> pd.DataFrame:
        if self._transformer is not None:
            return pd.DataFrame(
                data = self._transformer.inverse_transform(data.values),
                columns = data.columns,
                index = data.index
            )
        return data

    def annotate_metadata(self,
                          data: pd.DataFrame,
                          file_name: str) -> pd.DataFrame:

        ref_value = self._metadata.loc[
            self._metadata[self._sample_identifier_column] == file_name,
            self._reference_column
        ].iloc[0]

        batch_value = self._metadata.loc[
            self._metadata[self._sample_identifier_column] == file_name,
            self._batch_column
        ].iloc[0]

        data[self._reference_column] = ref_value
        data[self._batch_column] = batch_value
        data[self._sample_identifier_column] = file_name

        data = data.set_index(
            [
                self._reference_column,
                self._batch_column,
                self._sample_identifier_column
            ]
        )

        return data


class DataProviderFCS(DataProvider):

    def __init__(self,
                 input_directory: PathLike,
                 truncate_max_range: bool,
                 sample_identifier_column: str,
                 reference_column: str,
                 batch_column: str,
                 metadata: pd.DataFrame,
                 channels: Optional[list[str]],
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
        data = self._reader.parse_fcs_df(file_name)
        data = self.annotate_metadata(data, file_name)
        data = self.select_channels(data)
        data = self.transform_data(data)
        return data


class DataProviderAnnData(DataProvider):

    def __init__(self,
                 adata: AnnData,
                 layer: str,
                 sample_identifier_column: str,
                 reference_column: str,
                 batch_column: str,
                 metadata: pd.DataFrame,
                 channels: Optional[list[str]],
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
                         file_name: str):
        return self._adata[
            self._adata.obs[self._sample_identifier_column] == file_name,
            :
        ].to_df(layer = self._layer)

    def prep_dataframe(self,
                       file_name: str) -> pd.DataFrame:
        data = self.parse_anndata_df(file_name)
        data = self.annotate_metadata(data, file_name)
        data = self.select_channels(data)
        data = self.transform_data(data)
        return data

