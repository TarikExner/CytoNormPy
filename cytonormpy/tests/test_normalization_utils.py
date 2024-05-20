import pytest
import pandas as pd

from cytonormpy._utils._utils import (_all_batches_have_reference)


def test_all_batches_have_reference():
    ref = ["control", "other", "control", "other", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )

    assert _all_batches_have_reference(df,
                                       "reference",
                                       "batch",
                                       ref_control_value = "control")


def test_all_batches_have_reference_ValueError():
    ref = ["control", "other", "control", "unknown", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    with pytest.raises(ValueError):
        _all_batches_have_reference(df,
                                    "reference",
                                    "batch",
                                    ref_control_value = "control")


def test_all_batches_have_reference_batch_only_controls():
    ref = ["control", "other", "control", "control", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    assert _all_batches_have_reference(df,
                                       "reference",
                                       "batch",
                                       ref_control_value = "control")


def test_all_batches_have_reference_batch_false():
    ref = ["control", "other", "other", "other", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    assert not _all_batches_have_reference(df,
                                           "reference",
                                           "batch",
                                           ref_control_value = "control")


def test_all_batches_have_reference_batch_wrong_control_value():
    ref = ["control", "other", "other", "other", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    assert not _all_batches_have_reference(df,
                                           "reference",
                                           "batch",
                                           ref_control_value = "ref")

