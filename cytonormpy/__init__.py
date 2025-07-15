import sys
from ._cytonorm import CytoNorm, example_cytonorm, example_anndata
from ._dataset import FCSFile
from ._clustering import FlowSOM, KMeans, MeanShift, AffinityPropagation
from ._transformation import (
    AsinhTransformer,
    HyperLogTransformer,
    LogTransformer,
    LogicleTransformer,
    Transformer,
)
from ._cytonorm import read_model
from ._evaluation import (
    mad_from_fcs,
    mad_comparison_from_fcs,
    mad_from_anndata,
    mad_comparison_from_anndata,
    emd_from_fcs,
    emd_comparison_from_fcs,
    emd_from_anndata,
    emd_comparison_from_anndata,
)
from . import _plotting as pl
from ._plotting import scatter, histogram, emd, mad, cv_heatmap, splineplot, Plotter

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pl"]})

__all__ = [
    "CytoNorm",
    "FlowSOM",
    "KMeans",
    "MeanShift",
    "AffinityPropagation",
    "example_anndata",
    "example_cytonorm",
    "Transformer",
    "AsinhTransformer",
    "HyperLogTransformer",
    "LogTransformer",
    "LogicleTransformer",
    "FCSFile",
    "read_model",
    "mad_from_fcs",
    "mad_comparison_from_fcs",
    "mad_from_anndata",
    "mad_comparison_from_anndata",
    "emd_from_fcs",
    "emd_comparison_from_fcs",
    "emd_from_anndata",
    "emd_comparison_from_anndata",
    "pl",
    "scatter",
    "histogram",
    "emd",
    "mad",
    "cv_heatmap",
    "splineplot",
    "Plotter",
]

__version__ = "1.0.2"
