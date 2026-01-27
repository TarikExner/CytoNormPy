import pytest
import numpy as np
from anndata import AnnData
from cytonormpy import CytoNorm


def test_layer_none_uses_adata_x(data_anndata):
    """Test that layer=None uses AnnData.X directly instead of a named layer."""
    # Create a new AnnData with data in .X instead of a layer
    adata = data_anndata.copy()

    # Move data from 'compensated' layer to .X
    adata.X = adata.layers["compensated"].copy()

    # Remove the layer so we're only working with .X
    del adata.layers["compensated"]

    # Verify data is in .X
    assert adata.X is not None
    assert "compensated" not in adata.layers

    # This should work with layer=None
    cn = CytoNorm()
    cn.run_anndata_setup(
        adata=adata,
        layer=None,  # Use .X instead of a named layer
        reference_column="reference",
        reference_value="ref",
        batch_column="batch",
        sample_identifier_column="file_name",
        channels="markers",
    )

    # Verify the setup worked
    assert hasattr(cn, "_datahandler")
    assert "cyto_normalized" in adata.layers

    # Run normalization workflow
    cn.calculate_quantiles()
    cn.calculate_splines()
    cn.normalize_data()

    # Verify normalized data exists and is different from original
    assert "cyto_normalized" in adata.layers
    assert not np.array_equal(adata.X, adata.layers["cyto_normalized"])


def test_layer_none_end_to_end(data_anndata):
    """Test full normalization workflow with layer=None."""
    adata = data_anndata.copy()

    # Move data to .X
    adata.X = adata.layers["compensated"].copy()
    del adata.layers["compensated"]

    cn = CytoNorm()
    cn.run_anndata_setup(
        adata=adata,
        layer=None,
        reference_column="reference",
        reference_value="ref",
        batch_column="batch",
        sample_identifier_column="file_name",
        channels="markers",
    )

    cn.calculate_quantiles()
    cn.calculate_splines()

    # Normalize validation samples first
    val_file_names = adata.obs[adata.obs["reference"] == "other"]["file_name"].unique().tolist()
    batches = [
        adata.obs.loc[adata.obs["file_name"] == file, "batch"].unique().tolist()[0]
        for file in val_file_names
    ]
    cn.normalize_data(file_names=val_file_names, batches=batches)

    # Verify normalization worked
    assert "cyto_normalized" in adata.layers

    # Reference files should be unchanged (same as original in .X)
    ref_mask = adata.obs["reference"] == "ref"
    assert np.array_equal(
        adata[ref_mask].X,
        adata[ref_mask].layers["cyto_normalized"],
    )


def test_layer_none_evaluation_functions(data_anndata):
    """Test that evaluation functions work with layer=None."""
    adata = data_anndata.copy()

    # Move data to .X
    adata.X = adata.layers["compensated"].copy()
    del adata.layers["compensated"]

    cn = CytoNorm()
    cn.run_anndata_setup(
        adata=adata,
        layer=None,
        reference_column="reference",
        reference_value="ref",
        batch_column="batch",
        sample_identifier_column="file_name",
        channels="markers",
    )

    cn.calculate_quantiles()
    cn.calculate_splines()
    cn.normalize_data()

    # Test MAD calculation (requires both original and normalized layers)
    # Since original data is in .X, we need to pass layer=None for original
    cn.calculate_mad()
    assert cn.mad_frame is not None
    assert len(cn.mad_frame) > 0

    # Test EMD calculation
    cn.calculate_emd()
    assert cn.emd_frame is not None
    assert len(cn.emd_frame) > 0
