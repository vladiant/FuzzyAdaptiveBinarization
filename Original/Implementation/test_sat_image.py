import pytest
import numpy as np

from base_impl import fuzzy_sat

test_image = np.array(
    [
        [
            2.92003782e-02,
            9.73964043e-01,
            6.31460529e-01,
            7.91803968e-01,
            4.66192373e-01,
            7.75964369e-01,
            5.35925355e-01,
            7.42816952e-01,
        ],
        [
            8.39648084e-01,
            5.64459660e-02,
            9.99594686e-01,
            5.25247054e-02,
            2.06084719e-01,
            9.69003894e-01,
            6.69672643e-01,
            5.03201508e-01,
        ],
        [
            6.45017105e-01,
            9.99368827e-01,
            1.87596326e-01,
            1.17496388e-01,
            6.31967362e-01,
            6.28218244e-01,
            2.09527893e-02,
            5.59388553e-02,
        ],
        [
            6.08544141e-01,
            7.96379594e-01,
            6.33429452e-02,
            5.28422655e-01,
            9.17656014e-01,
            6.33443203e-01,
            1.18714017e-01,
            8.74972660e-01,
        ],
        [
            7.72973543e-01,
            2.99287927e-01,
            6.61012884e-01,
            8.60757470e-01,
            9.89214476e-01,
            2.93357811e-01,
            9.30261976e-01,
            6.75526630e-01,
        ],
        [
            2.05553733e-01,
            3.58409278e-01,
            7.63378032e-01,
            1.24160785e-01,
            5.34945024e-01,
            8.37360743e-02,
            7.03380188e-01,
            7.97952046e-01,
        ],
        [
            3.44256383e-01,
            8.36403050e-01,
            9.85031216e-01,
            9.80954108e-01,
            7.73170538e-01,
            9.46561977e-01,
            8.18915411e-04,
            3.41162441e-01,
        ],
        [
            5.88494902e-01,
            5.29670290e-01,
            2.92132969e-01,
            9.47695627e-02,
            7.99589162e-01,
            3.26008299e-01,
            1.68391820e-03,
            3.75414636e-01,
        ],
    ]
)

expected_s = np.array(
    [
        [
            2.92003788e-02,
            1.00316441e00,
            1.63462496e00,
            2.42642879e00,
            2.89262104e00,
            3.66858530e00,
            4.20451069e00,
            4.94732761e00,
        ],
        [
            8.68848443e-01,
            1.89925838e00,
            3.53031349e00,
            4.37464190e00,
            5.04691887e00,
            6.79188633e00,
            7.99748373e00,
            9.24350166e00,
        ],
        [
            1.51386547e00,
            3.54364443e00,
            5.36229610e00,
            6.32412148e00,
            7.62836599e00,
            1.00015516e01,
            1.12281027e01,
            1.25300579e01,
        ],
        [
            2.12240958e00,
            4.94856834e00,
            6.83056307e00,
            8.32081127e00,
            1.05427113e01,
            1.35493393e01,
            1.48946047e01,
            1.70715332e01,
        ],
        [
            2.89538312e00,
            6.02083015e00,
            8.56383705e00,
            1.09148445e01,
            1.41259584e01,
            1.74259434e01,
            1.97014694e01,
            2.25539246e01,
        ],
        [
            3.10093689e00,
            6.58479404e00,
            9.89117908e00,
            1.23663464e01,
            1.61124039e01,
            1.94961281e01,
            2.24750347e01,
            2.61254425e01,
        ],
        [
            3.44519329e00,
            7.76545334e00,
            1.20568686e01,
            1.55129900e01,
            2.00322208e01,
            2.43625069e01,
            2.73422318e01,
            3.13338032e01,
        ],
        [
            4.03368807e00,
            8.88361835e00,
            1.34671669e01,
            1.70180588e01,
            2.23368797e01,
            2.69931736e01,
            2.99745827e01,
            3.43415680e01,
        ],
    ]
)


def test_compute_sat_s_image():
    int_img = fuzzy_sat(test_image)

    int_img.compute_sat()

    test_s = int_img.get_S()
    np.testing.assert_allclose(expected_s, test_s)
