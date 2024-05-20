
from anndata import AnnData
import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

from typing import Optional, Literal, Union





def cytonorm(adata: AnnData,
             reference: str = "batch_reference",
             batch: str = "batch",
             cluster_key: Optional[str] = None,
             cluster_algorithm: Literal["FlowSOM"] = "FlowSOM",
             cluster_kwargs: Optional[dict] = None,
             cv_cutoff: float = 2,
             sample_identifier: str = "sample_ID",
             key_added: str = "normalized",
             copy: bool = False) -> Optional[AnnData]:
    """\
    Applies the CytoNorm algorithm.

    Parameters
    ----------

    adata
        The AnnData object
    reference
        The column in `adata.obs` that specifies whether a sample
        is used for reference and is therefore present in all batches.
    batch
        The column in `adata.obs` that specifies the batch.
    cluster_key
        If clusters have been calculated, specify the column in `adata.obs`
        that stores the clustering information
    cluster_algorithm
        Choose the clustering algorithm to determine the clusters for
        normalization. Currently, only `FlowSOM` is implemented.
    cluster_kwargs
        Dictionary of keyword arguments passed to the clustering algorithm.
    sample_identifier
        The column in `adata.obs` where the individual samples are identified.
        Defaults to 'sample_ID'.
    cv_cutoff
        Specifying the cutoff of CV values per cluster. A high CV value
        indicates that cluster identity is dependent on the batch. In that
        case, normalization by clusters is inappropriate. Defaults to 2.
    key_added
        Name for the layer in `adata.layers` for the normalized data.

    Returns
    -------

    The anndata object with the normalized values stored in
    adata.layers[{key_added}] or None, depending on `copy`.


    """

    if cluster_kwargs is None:
        cluster_kwargs = {}

    if cluster_algorithm != "FlowSOM":
        raise NotImplementedError(
            f"Clustering algorithm has to be FlowSOM, was {cluster_algorithm}."
        )

    assert _all_batches_have_reference(adata.obs, reference, batch)

    train_adata = adata[adata.obs[reference] == REFERENCE_CONTROL_VALUE]
    validation_adata = adata[adata.obs[reference] != REFERENCE_CONTROL_VALUE]

    if cluster_key is None:
        _calculate_flowsom(train_adata,
                           **cluster_kwargs)
        cluster_key = "clusters"

    if not _all_cvs_below_cutoff(train_adata,
                                 cluster_key = cluster_key,
                                 sample_key = sample_identifier,
                                 cv_cutoff = cv_cutoff):
        raise ValueError(
            "CVs were too high. Its not appropriate to use "
            "cluster-based normalization!"
        )
    # groupby sample_ID in train_data // batch, get quantiles per group






