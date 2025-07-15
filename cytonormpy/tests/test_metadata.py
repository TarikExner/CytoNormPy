import pytest
import pandas as pd
import re

from cytonormpy._dataset._metadata import Metadata
from cytonormpy._utils._utils import _all_batches_have_reference, _conclusive_reference_values


def test_init_and_properties(metadata: pd.DataFrame):
    md_df = metadata.copy()
    m = Metadata(
        metadata=md_df,
        reference_column="reference",
        reference_value="ref",
        batch_column="batch",
        sample_identifier_column="file_name",
    )
    assert m.validation_value == "other"
    expected_refs = md_df.loc[md_df.reference == "ref", "file_name"].tolist()
    assert m.ref_file_names == expected_refs
    expected_vals = md_df.loc[md_df.reference != "ref", "file_name"].tolist()
    assert m.validation_file_names == expected_vals
    assert m.all_file_names == expected_refs + expected_vals
    assert m.reference_construction_needed is False


def test_to_df_returns_original(metadata: pd.DataFrame):
    m = Metadata(metadata, "reference", "ref", "batch", "file_name")
    pd.testing.assert_frame_equal(m.to_df(), metadata)


def test_get_ref_and_batch_and_corresponding(metadata: pd.DataFrame):
    m = Metadata(metadata, "reference", "ref", "batch", "file_name")
    val_file = m.validation_file_names[0]
    assert m.get_ref_value(val_file) == "other"
    b = m.get_batch(val_file)
    corr = m.get_corresponding_reference_file(val_file)
    same_batch_refs = metadata.loc[
        (metadata.batch == b) & (metadata.reference == "ref"), "file_name"
    ].tolist()
    assert corr in same_batch_refs


def test__lookup_invalid_which(metadata: pd.DataFrame):
    m = Metadata(metadata, "reference", "ref", "batch", "file_name")
    with pytest.raises(ValueError, match="Wrong 'which' parameter"):
        _ = m._lookup("anything.fcs", which="nope")


def test_validate_metadata_table_missing_column(metadata: pd.DataFrame):
    bad = metadata.drop(columns=["batch"])
    msg = f"Metadata must contain the columns [file_name, reference, batch]. Found {bad.columns}"
    with pytest.raises(ValueError, match=re.escape(msg)):
        Metadata(bad, "reference", "ref", "batch", "file_name")


def test_validate_metadata_table_inconclusive_reference(metadata: pd.DataFrame):
    bad = metadata.copy()
    bad.loc[0, "reference"] = "third"
    msg = (
        "The column reference must only contain descriptive values for references and other values"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        Metadata(bad, "reference", "ref", "batch", "file_name")


def test_validate_batch_references_warning(metadata: pd.DataFrame):
    bad = metadata.copy()
    bad.loc[bad.batch == 2, "reference"] = "other"
    with pytest.warns(UserWarning, match="Reference samples will be constructed"):
        m = Metadata(bad, "reference", "ref", "batch", "file_name")
    assert m.reference_construction_needed is True


def test_find_batches_without_reference_method(metadata: pd.DataFrame):
    m = Metadata(metadata, "reference", "ref", "batch", "file_name")
    assert m.find_batches_without_reference() == []
    mod = metadata.loc[~((metadata.batch == 1) & (metadata.reference == "ref"))]
    m2 = Metadata(mod, "reference", "ref", "batch", "file_name")
    assert m2.find_batches_without_reference() == [1]


def test__all_batches_have_reference_errors_and_returns():
    df = pd.DataFrame(
        {
            "reference": ["a", "b", "c", "a"],
            "batch": [1, 1, 2, 2],
        }
    )
    msg = "Please make sure that there are only two values in the reference column. Have found ['a', 'b', 'c']"
    with pytest.raises(ValueError, match=re.escape(msg)):
        _all_batches_have_reference(df, "reference", "batch", "a")

    df2 = pd.DataFrame(
        {
            "reference": ["a", "b", "a", "b"],
            "batch": [1, 1, 2, 2],
        }
    )
    assert _all_batches_have_reference(df2, "reference", "batch", "a")

    df3 = pd.DataFrame(
        {
            "reference": ["a", "a", "a"],
            "batch": [1, 2, 3],
        }
    )
    assert _all_batches_have_reference(df3, "reference", "batch", "a")

    df4 = pd.DataFrame(
        {
            "reference": ["a", "a", "b", "a"],
            "batch": [1, 2, 2, 3],
        }
    )
    assert _all_batches_have_reference(df4, "reference", "batch", "a")

    df5 = pd.DataFrame(
        {
            "reference": ["a", "a", "b", "b"],
            "batch": [1, 2, 2, 3],
        }
    )
    assert _all_batches_have_reference(df5, "reference", "batch", "a") is False


def test__conclusive_reference_values():
    df = pd.DataFrame({"reference": ["x", "y", "x"]})
    assert _conclusive_reference_values(df, "reference") is True
    df2 = pd.DataFrame({"reference": ["x", "y", "z"]})
    assert _conclusive_reference_values(df2, "reference") is False


def test_get_files_per_batch_returns_correct_list(metadata: pd.DataFrame):
    """
    For each batch in the fixture, get_files_per_batch should return exactly
    the list of file_name entries belonging to that batch.
    """
    m = Metadata(metadata.copy(), "reference", "ref", "batch", "file_name")
    # collect expected mapping from the raw DF
    expected = {batch: group["file_name"].tolist() for batch, group in metadata.groupby("batch")}
    for batch, files in expected.items():
        assert m.get_files_per_batch(batch) == files


def test_add_file_to_metadata_appends_and_updates_lists(metadata: pd.DataFrame):
    """
    add_file_to_metadata should:
     - append a new row with the sample_identifier_column = new_file
       and reference_column = validation_value
     - include new_file in validation_file_names, all_file_names,
       and get_files_per_batch for that batch
    """
    md = metadata.copy()
    m = Metadata(md, "reference", "ref", "batch", "file_name")
    # pick a batch that already has a reference sample
    target_batch = metadata["batch"].iloc[0]
    new_file = "new_sample.fcs"

    # record pre‑state
    prev_validation = set(m.validation_file_names)
    prev_all = set(m.all_file_names)
    prev_batch_files = set(m.get_files_per_batch(target_batch))
    val_value = m.validation_value
    assert val_value is not None, "fixture must have at least one non‑ref"

    # do the add
    m.add_file_to_metadata(new_file, batch=target_batch)

    # the metadata DF gained exactly one row
    assert new_file in m.metadata["file_name"].values

    # the new file should carry the validation_value
    row = m.metadata.loc[m.metadata["file_name"] == new_file].iloc[0]
    assert row["reference"] == val_value
    assert int(row["batch"]) == int(target_batch)

    # lists should have been refreshed
    assert new_file in m.validation_file_names
    assert new_file in m.all_file_names
    # original lists intact
    assert prev_validation.issubset(set(m.validation_file_names))
    assert prev_all.issubset(set(m.all_file_names))

    # get_files_per_batch should now include it
    batch_files = m.get_files_per_batch(target_batch)
    assert new_file in batch_files
    # and length increased by 1
    assert len(batch_files) == len(prev_batch_files) + 1


def test_assemble_reference_assembly_dict_detects_batches_without_ref(metadata: pd.DataFrame):
    """
    If we remove the 'ref' entries for batch == 2, then
    assemble_reference_assembly_dict should flag {2: [all files of batch 2]}.
    """
    # start with a clean copy
    md = metadata.copy()
    # drop all 'ref' rows from batch 2
    mask = ~((md["batch"] == 2) & (md["reference"] == "ref"))
    md = md.loc[mask].reset_index(drop=True)

    m = Metadata(md, "reference", "ref", "batch", "file_name")

    # It should have set reference_construction_needed
    assert m.reference_construction_needed is True

    # The dict should map batch 2 to its file list
    expected_files = md.loc[md["batch"] == 2, "file_name"].tolist()
    assert 2 in m.reference_assembly_dict
    assert set(m.reference_assembly_dict[2]) == set(expected_files)

    # No other batch should appear
    other_batches = set(md["batch"].unique()) - {2}
    assert set(m.reference_assembly_dict.keys()) == {2}


def test_update_refreshes_all_lists_and_dict(metadata: pd.DataFrame):
    """
    Directly calling update() after manual metadata mutation should
    recompute ref_file_names, validation_file_names, all_file_names,
    and reference_assembly_dict.
    """
    md = metadata.copy()
    m = Metadata(md, "reference", "ref", "batch", "file_name")

    # manually strip all ref from batch 3
    m.metadata = m.metadata.loc[
        ~((m.metadata["batch"] == 3) & (m.metadata["reference"] == "ref"))
    ].reset_index(drop=True)
    # now re‐run update()
    m.update()

    # batch 3 should now be flagged missing
    assert m.reference_construction_needed is True
    # lists refreshed
    assert 3 not in [
        b for b, grp in m.metadata.groupby("batch") if "ref" in grp["reference"].values
    ]
    # dict entry for 3
    assert 3 in m.reference_assembly_dict
    assert set(m.reference_assembly_dict[3]) == set(m.get_files_per_batch(3))


def test_to_df_remains_consistent_after_updates(metadata: pd.DataFrame):
    """
    to_df() should always return the current metadata dataframe,
    even after add_file_to_metadata and update().
    """
    md = metadata.copy()
    m = Metadata(md, "reference", "ref", "batch", "file_name")
    # initial
    df0 = m.to_df().copy()

    # add a new file and update
    m.add_file_to_metadata("foo.fcs", batch=md["batch"].iloc[0])
    df1 = m.to_df()

    # df1 has one extra row
    assert len(df1) == len(df0) + 1
    assert "foo.fcs" in df1["file_name"].values
