from cytonormpy._normalization import Spline
from cytonormpy._normalization._spline_calc import IdentitySpline

import numpy as np


def test_spline_func():
    # we want to test if the R-function and the
    # python equivalent behave similarly.

    x = np.array([1, 4, 6, 12, 17, 20], dtype = np.float64)
    y = np.array([0.7, 4.5, 8.2, 11.4, 17, 21.2], dtype = np.float64)

    s = Spline(
        batch = 1,
        channel = "BV421-A",
        cluster = 4
    )
    s.fit(x, y)

    test_arr = np.arange(-2, 25) + 0.5  # we deliberately go outside the range
    res = s.transform(test_arr)

    # R code:
    # x = c(1, 4, 6, 12, 17, 20)
    # y = c(0.7, 4.5, 8.2, 11.4, 17, 21.2)
    # spl = stats::splinefun(x, y, method = "monoH.FC")
    # spl(seq(-2, 24)+0.5)

    r_array = np.array([
        -2.46666667, -1.20000000, 0.06666667, 1.31307870, 2.49062500,
        3.76539352, 5.40468750, 7.43281250, 8.73205440, 9.47296875, 9.91513310,
        10.21715856, 10.53765625, 11.03523727, 11.83490000, 12.82030000,
        13.92916667, 15.12470000, 16.37010000, 17.65138889, 19.04750000,
        20.49027778, 21.90000000, 23.30000000, 24.70000000, 26.10000000,
        27.50000000
    ])

    np.testing.assert_array_almost_equal(res, r_array, decimal = 6)


def test_identity_func():
    x = np.array([1, 4, 6, 12, 17, 20])
    y = np.array([0.7, 4.5, 8.2, 11.4, 17, 21.2])

    s = Spline(batch = 1,
               channel = "BV421-A",
               cluster = 4,
               spline_calc_function = IdentitySpline)
    s.fit(x, y)
    test_arr = np.arange(-2, 25) + 0.5  # we deliberately go outside the range

    assert np.array_equal(test_arr, s.transform(test_arr))
