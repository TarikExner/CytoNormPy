import pandas as pd
from cytonormpy._dataset._datareader import DataReaderFCS
from cytonormpy import FCSFile

def test_fcs_reading_fcsfile(INPUT_DIR: str,
                             metadata: pd.DataFrame):
    reader = DataReaderFCS(input_directory = INPUT_DIR)
    file_names = metadata["file_name"].tolist()
    data = reader.parse_fcs_file(file_names[0])

    assert isinstance(data, FCSFile)


def test_fcs_reading_dataframe(INPUT_DIR: str,
                               metadata: pd.DataFrame):
    reader = DataReaderFCS(input_directory = INPUT_DIR)
    file_names = metadata["file_name"].tolist()
    data = reader.parse_fcs_df(file_names[0])

    assert isinstance(data, pd.DataFrame)

