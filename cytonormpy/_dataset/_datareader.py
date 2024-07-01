from os import PathLike
import pandas as pd

from typing import Union

from ._fcs_file import FCSFile


class DataReader:

    def __init__(self):
        pass

class DataReaderFCS(DataReader):
    """\
    Class to handle the data reading from disk for FCS files.
    This class returns the raw data as read from disk when
    the method `.parse_fcs_file()` is used. Use `parse_fcs_df()`
    to get a :class:`pandas.DataFrame`.

    Parameters
    ----------
    input_directory
        The input directory where the FCS files are stored
    truncate_max_range
        Whether to truncate the values of the FCS file to
        the documented PnR values.

    Returns
    -------
    None

    """
    def __init__(self,
                 input_directory: Union[PathLike, str],
                 truncate_max_range: bool = True):
        self._input_dir = input_directory
        self._truncate_max_range = truncate_max_range
    
    def parse_fcs_df(self,
                     file_name: str) -> pd.DataFrame:
        """\
        Reads an FCS file and creates a dataframe where
        the columns represent the channels and the rows
        correspond to the individual cells.

        Parameters
        ----------
        file_name
            The file name of the FCS file to be read.

        Returns
        -------
        A :class:`pandas.DataFrame`
        """

        return self.parse_fcs_file(file_name = file_name).to_df()

    def parse_fcs_file(self,
                       file_name: str) -> FCSFile:
        """\
        Reads an FCS File from disk and provides it as an
        FCSFile instance.

        Parameters
        ----------
        file_name
            The file name of the FCS file to be read.

        Returns
        -------
        A :class:`cytonormpy.FCSFile`
        """
        return FCSFile(
            input_directory = self._input_dir,
            file_name = file_name,
            truncate_max_range = self._truncate_max_range
        )

class DataReaderAnnData(DataReader):

    def __init__(self):
        pass
