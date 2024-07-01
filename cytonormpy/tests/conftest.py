import pytest
import anndata as ad
from anndata import AnnData
import numpy as np
import pandas as pd
import os
from pathlib import Path

from cytonormpy._dataset._fcs_file import FCSFile
from cytonormpy._dataset._dataset import DataHandlerAnnData, DataHandlerFCS


@pytest.fixture
def DATAHANDLER_DEFAULT_KWARGS():
    return {
        "layer": "compensated",
        "reference_column": "reference",
        "reference_value": "ref",
        "batch_column": "batch",
        "sample_identifier_column": "file_name",
        "channels": "markers"
    }


@pytest.fixture
def INPUT_DIR():
    return Path(os.path.join(Path(__file__).parent.parent, "_resources"))


@pytest.fixture
def metadata() -> pd.DataFrame:
    HERE = Path(__file__).parent
    pkg_folder = HERE.parent
    return pd.read_csv(os.path.join(pkg_folder, "_resources/metadata_sid.csv"))


@pytest.fixture
def detectors() -> list[str]:
    return [
        'Y89Di', 'Pd102Di', 'Pd104Di', 'Pd105Di', 'Pd106Di', 'Pd108Di',
        'In113Di', 'In115Di', 'I127Di', 'Ba138Di', 'La139Di', 'Ce140Di',
        'Pr141Di', 'Nd142Di', 'Nd143Di', 'Nd144Di', 'Nd145Di', 'Nd146Di',
        'Sm147Di', 'Nd148Di', 'Sm149Di', 'Sm150Di', 'Eu151Di', 'Sm152Di',
        'Eu153Di', 'Sm154Di', 'Gd155Di', 'Gd156Di', 'Gd157Di', 'Gd158Di',
        'Tb159Di', 'Gd160Di', 'Dy161Di', 'Dy162Di', 'Dy163Di', 'Dy164Di',
        'Ho165Di', 'Er166Di', 'Er167Di', 'Er168Di', 'Tm169Di', 'Er170Di',
        'Yb171Di', 'Yb172Di', 'Yb173Di', 'Yb174Di', 'Lu175Di', 'Yb176Di',
        'Ir191Di', 'Ir193Di', 'Pt195Di', 'beadDist', 'Pd110Di', 'Time'
        'Event_length'
    ]



@pytest.fixture
def data_anndata() -> AnnData:
    HERE = Path(__file__).parent
    pkg_folder = HERE.parent
    fcs_dir = os.path.join(pkg_folder, "_resources")
    adata_file = Path(os.path.join(fcs_dir, "test_adata.h5ad"))

    if os.path.isfile(adata_file):
        return ad.read_h5ad(adata_file)

    fcs_files = [file for file in os.listdir(fcs_dir)
                 if file.endswith(".fcs")]
    adatas = []
    metadata = pd.read_csv(os.path.join(fcs_dir, "metadata_sid.csv"))
    for file in fcs_files:
        fcs = FCSFile(input_directory = fcs_dir,
                      file_name = file)
        events = fcs.original_events
        md_row = metadata.loc[
            metadata["file_name"] == file, :
        ].to_numpy()
        obs = np.repeat(
            md_row,
            events.shape[0],
            axis = 0
        )
        var_frame = fcs.channels
        obs_frame = pd.DataFrame(
            data = obs,
            columns = metadata.columns,
            index = pd.Index([str(i) for i in range(events.shape[0])])
        )
        adata = ad.AnnData(
            obs = obs_frame,
            var = var_frame,
            layers = {"compensated": events}
        )
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adatas.append(adata)

    dataset = ad.concat(adatas, axis = 0, join = "outer", merge = "same")
    dataset.var_names_make_unique()
    dataset.obs_names_make_unique()

    dataset.write(adata_file)
    return dataset

@pytest.fixture
def datahandleranndata(data_anndata: AnnData,
                       DATAHANDLER_DEFAULT_KWARGS: dict) -> DataHandlerAnnData:
    return DataHandlerAnnData(data_anndata, **DATAHANDLER_DEFAULT_KWARGS)


@pytest.fixture
def datahandlerfcs(metadata: pd.DataFrame,
                   INPUT_DIR: Path) -> DataHandlerFCS:
    return DataHandlerFCS(metadata = metadata,
                          input_directory = INPUT_DIR)

@pytest.fixture
def array_data(datahandleranndata: DataHandlerAnnData) -> np.ndarray:
    return datahandleranndata.ref_data_df.to_numpy()


