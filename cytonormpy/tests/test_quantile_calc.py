import pytest
import numpy as np
from cytonormpy._normalization._quantile_calc import ExpressionQuantiles


N_QUANTILES = 5
N_CHANNELS = 4
N_CLUSTERS = 6
N_BATCHES = 3


@pytest.fixture
def expr_q():
    return ExpressionQuantiles(
        n_batches = N_BATCHES,
        n_channels = N_CHANNELS,
        n_quantiles = N_QUANTILES,
        n_clusters = N_CLUSTERS
    )


def test_quantile_init(expr_q: ExpressionQuantiles):
    q_arr = expr_q.quantiles
    assert isinstance(q_arr, np.ndarray)
    assert q_arr.shape[0] == N_QUANTILES
    assert np.max(q_arr) == 1
    assert np.min(q_arr) == 0


def test_storage_array_init(expr_q: ExpressionQuantiles):
    arr = expr_q._expr_quantiles
    assert arr.shape == (5, 4, 6, 3)
    assert np.sum(arr) == 0


def test_quantile_calculation(expr_q: ExpressionQuantiles):
    test_arr = np.arange(101, dtype = np.float64).reshape(101, 1)
    res = expr_q.calculate_quantiles(test_arr)
    assert res.ndim == 4
    assert res.shape[0] == N_QUANTILES
    assert np.array_equal(res.flatten(),
                          np.array([0, 25, 50, 75, 100]))


def test_add_quantiles(expr_q: ExpressionQuantiles):
    data_array = np.random.randint(0, 100, N_CHANNELS * 20).reshape(20, N_CHANNELS).astype(np.float64)
    q = np.quantile(data_array, expr_q.quantiles, axis = 0)
    q = q[:, :, np.newaxis, np.newaxis]
    expr_q.add_quantiles(q, batch_idx = 2, cluster_idx = 1)

    assert np.array_equal(
        expr_q.get_quantiles(batch_idx = 2,
                             cluster_idx = 1,
                             flattened = False),
        q
    )
    assert np.array_equal(
        expr_q._expr_quantiles[:, :, 1, 2][:, :, np.newaxis, np.newaxis],
        q
    )


def test_add_nan_slice(expr_q: ExpressionQuantiles):
    expr_q.add_nan_slice(batch_idx = 1,
                         cluster_idx = 2)
    assert np.all(
        np.isnan(
            expr_q.get_quantiles(batch_idx = 1, cluster_idx = 2)
        )
    )

    assert expr_q._is_nan_slice(
        expr_q.get_quantiles(batch_idx = 1, cluster_idx = 2)
    )
