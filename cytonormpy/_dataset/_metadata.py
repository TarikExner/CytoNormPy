import numpy as np
import pandas as pd
import warnings

from typing import Literal, Union

from pandas.api.types import is_numeric_dtype

from .._utils._utils import _all_batches_have_reference, _conclusive_reference_values


class Metadata:
    def __init__(
        self,
        metadata: pd.DataFrame,
        reference_column: str,
        reference_value: str,
        batch_column: str,
        sample_identifier_column: str,
    ) -> None:
        self.metadata = metadata
        self.reference_column = reference_column
        self.reference_value = reference_value
        self.batch_column = batch_column
        self.sample_identifier_column = sample_identifier_column

        self.reference_construction_needed = False

        self.update()

        try:
            self.validation_value = list(
                set(
                    [
                        val
                        for val in self.metadata[self.reference_column]
                        if val != self.reference_value
                    ]
                )
            )[0]
        except IndexError:  # means we only have reference values
            self.validation_value = None

    def update(self):
        self.validate_metadata()

        self.ref_file_names = self.get_reference_file_names()
        self.validation_file_names = self.get_validation_file_names()
        self.all_file_names = self.ref_file_names + self.validation_file_names

        self.assemble_reference_assembly_dict()

    def validate_metadata(self) -> None:
        self.validate_metadata_table()
        self.validate_batch_references()
        self.convert_batch_dtype()

    def to_df(self) -> pd.DataFrame:
        return self.metadata

    def get_reference_file_names(self) -> list[str]:
        return (
            self.metadata.loc[
                self.metadata[self.reference_column] == self.reference_value,
                self.sample_identifier_column,
            ]
            .unique()
            .tolist()
        )

    def get_validation_file_names(self) -> list[str]:
        return (
            self.metadata.loc[
                self.metadata[self.reference_column] != self.reference_value,
                self.sample_identifier_column,
            ]
            .unique()
            .tolist()
        )

    def _lookup(
        self, file_name: str, which: Literal["batch", "reference_file", "reference_value"]
    ) -> str:
        if which == "batch":
            lookup_col = self.batch_column
        elif which == "reference_file":
            lookup_col = self.sample_identifier_column
        elif which == "reference_value":
            lookup_col = self.reference_column
        else:
            raise ValueError("Wrong 'which' parameter")
        return self.metadata.loc[
            self.metadata[self.sample_identifier_column] == file_name, lookup_col
        ].iloc[0]

    def get_ref_value(self, file_name: str) -> str:
        """Returns the corresponding reference value of a file."""
        return self._lookup(file_name, which="reference_value")

    def get_batch(self, file_name: str) -> str:
        """Returns the corresponding batch of a file."""
        return self._lookup(file_name, which="batch")

    def get_corresponding_reference_file(self, file_name) -> str:
        """Returns the corresponding reference file of a file."""
        batch = self.get_batch(file_name)
        return self.metadata.loc[
            (self.metadata[self.batch_column] == batch)
            & (self.metadata[self.reference_column] == self.reference_value),
            self.sample_identifier_column,
        ].iloc[0]

    def get_files_per_batch(self, batch) -> list[str]:
        return self.metadata.loc[
            self.metadata[self.batch_column] == batch, self.sample_identifier_column
        ].tolist()

    def add_file_to_metadata(self, file_name: str, batch: Union[str, int]) -> None:
        new_file_df = pd.DataFrame(
            data=[[file_name, self.validation_value, batch]],
            columns=[self.sample_identifier_column, self.reference_column, self.batch_column],
            index=[-1],
        )
        self.metadata = pd.concat([self.metadata, new_file_df], axis=0).reset_index(drop=True)
        self.update()

    def convert_batch_dtype(self) -> None:
        """
        If the batch is entered as a string, we convert them
        to integers in order to comply with the numpy sorts
        later on.
        """
        if not is_numeric_dtype(self.metadata[self.batch_column]):
            try:
                self.metadata[self.batch_column] = self.metadata[self.batch_column].astype(np.int8)
            except ValueError:
                self.metadata[f"original_{self.batch_column}"] = self.metadata[self.batch_column]
                mapping = {
                    entry: i for i, entry in enumerate(self.metadata[self.batch_column].unique())
                }
                self.metadata[self.batch_column] = self.metadata[self.batch_column].map(mapping)

    def validate_metadata_table(self):
        if not all(
            k in self.metadata.columns
            for k in [self.sample_identifier_column, self.reference_column, self.batch_column]
        ):
            raise ValueError(
                "Metadata must contain the columns "
                f"[{self.sample_identifier_column}, "
                f"{self.reference_column}, "
                f"{self.batch_column}]. "
                f"Found {self.metadata.columns}"
            )
        if not _conclusive_reference_values(self.metadata, self.reference_column):
            raise ValueError(
                f"The column {self.reference_column} must only contain "
                "descriptive values for references and other values"
            )

    def validate_batch_references(self):
        if not _all_batches_have_reference(
            self.metadata,
            reference=self.reference_column,
            batch=self.batch_column,
            ref_control_value=self.reference_value,
        ):
            self.reference_construction_needed = True
            warnings.warn("Reference samples will be constructed", UserWarning)

    def find_batches_without_reference(self):
        """
        Return a list of batch identifiers for which the given ref_control_value
        never appears in the reference column.
        """
        return [
            batch
            for batch, grp in self.metadata.groupby(self.batch_column)
            if self.reference_value not in grp[self.reference_column].values
        ]

    def assemble_reference_assembly_dict(self):
        """Builds a dictionary of shape {batch: [files, ...], ...} to store files of batches without references"""
        batches_wo_reference = self.find_batches_without_reference()
        self.reference_assembly_dict = {
            batch: self.get_files_per_batch(batch) for batch in batches_wo_reference
        }


class MockMetadata(Metadata):
    def __init__(self, sample_identifier_column: str) -> None:
        self.sample_identifier_column = sample_identifier_column
