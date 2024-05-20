import warnings
import numpy as np
import pandas as pd

from typing import Optional, Callable, Union
from functools import lru_cache, wraps


REFERENCE_CONTROL_VALUE = "control"
REFERENCE_VALIDATION_VALUE = "other"


class CustomArray:
    """\
    Makes numpy array hashable.
    """

    def __init__(self, x: np.ndarray) -> None:
        self.values = x
        # here you can use your own hashing function
        self.h = map(tuple, x)

    def __hash__(self) -> int:
        return hash(self.h)

    def __eq__(self, __value) -> bool:
        return __value.h == self.h


def np_cache(func):

    @lru_cache()
    def cached_wrapper(c_arr, *args, **kwargs):
        return func(c_arr.values, *args, **kwargs)

    @wraps(func)
    def wrapper(x: np.ndarray, *args, **kwargs):
        c_arr = CustomArray(x)
        return cached_wrapper(c_arr, *args, **kwargs)

    return wrapper


def regularize_values(x: np.ndarray,
                      y: np.ndarray,
                      ties: Optional[Union[str, Callable]] = np.mean
                      ) -> tuple[np.ndarray, np.ndarray]:
    """\
    Implementation of the R regularize.values function in python.
    Performance optimized as much as possible. 400 µs for 2 1000 random
    integer numpy arrays.
    """

    assert x.shape[0] == y.shape[0], "x and y length must be the same."

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

    nx = x.shape[0]

    if ties != "ordered":
        o = np.argsort(x)
        x = x[o]
        y = y[o]
        ux, idxs = np.unique(x, return_index = True)
        if ux.shape[0] < nx:
            if ties is None:
                warnings.warn(
                    "Collapsing to unique 'x' values",
                    UserWarning
                )
            assert isinstance(ties, Callable)
            # y = tapply(y, match(x, x), fun)
            ls, rs = match(x, x, sorter = np.argsort(x))
            matches = np.vstack([ls, rs]).T
            b = np.ascontiguousarray(matches)\
                .view(
                    np.dtype((
                        np.void,
                        matches.dtype.itemsize * matches.shape[1])
                    )
            )

            unique_matches = np.unique(b)\
                .view(matches.dtype)\
                .reshape(-1, matches.shape[1])
            unique_matches = np.sort(unique_matches, axis = 1)
            for b, e in zip(unique_matches[:, 0],
                            unique_matches[:, 1]):
                y = _insert_to_array(y, b, e, ties)

            x = x[idxs]
            y = y[idxs]

            assert x.shape[0] == y.shape[0]

    return x, y


@np_cache
def _insert_to_array(y, b, e, ties):
    y[b:e] = ties(y[b:e])
    return y


def match(x: np.ndarray,
          y: np.ndarray,
          sorter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.searchsorted(x, y, side = "left", sorter = sorter)
    right = np.searchsorted(x, y, side = "right", sorter = sorter)
    return left, right


def _all_batches_have_reference(df: pd.DataFrame,
                                reference: str,
                                batch: str,
                                ref_control_value: Optional[str]
                                ) -> bool:
    """
    Function checks if there are samples labeled ref_control_value
    for each batch.
    """
    _df: pd.DataFrame = pd.DataFrame(df[[reference, batch]].drop_duplicates())

    if len(_df[reference].unique()) != 2:
        raise ValueError(
            "Please make sure that there are only two values in "
            "the reference column. Have found "
            f"{_df[reference].unique().tolist()}"
        )

    # if both uniques are present in all batches, that's fine
    ref_per_batch = _df.groupby(batch, observed = True).nunique()
    if all(ref_per_batch[reference] == 2):
        return True

    # alternatively, batches might only contain controls
    one_refs = ref_per_batch[ref_per_batch[reference] == 1]
    one_ref_batches = one_refs.index.tolist()

    if all(
        _df.loc[
            _df[batch].isin(one_ref_batches), reference
        ] == ref_control_value
    ):
        return True

    return False


def _conclusive_reference_values(df: pd.DataFrame,
                                 reference: str) -> bool:
    """
    checks if there are no more than two values in the reference column.
    We allow the option that every sample is labeled as control.
    """
    if len(df[reference].unique()) > 2:
        return False
    return True
