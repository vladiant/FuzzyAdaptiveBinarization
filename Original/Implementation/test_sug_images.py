import pytest
import numpy as np

from base_impl import fuzzy_sat
from test_sat_image import test_image, expected_s

expected_sug_s_c = np.array(
    [
        [
            0.02920038,
            0.5,
            1.0031644,
            1.634625,
            2.4264288,
            2.892621,
            3.6685853,
            4.2045107,
        ],
        [0.5, 0.75, 1.0031644, 1.634625, 2.4264288, 2.892621, 3.6685853, 4.2045107],
        [
            0.86884844,
            0.86884844,
            1.8992584,
            3.5303135,
            4.374642,
            5.046919,
            6.7918863,
            7.9974837,
        ],
        [
            1.5138655,
            1.5138655,
            3.5436444,
            5.362296,
            6.3241215,
            7.628366,
            10.001552,
            11.228103,
        ],
        [
            2.1224096,
            2.1224096,
            4.9485683,
            6.830563,
            8.320811,
            10.542711,
            13.549339,
            14.894605,
        ],
        [
            2.8953831,
            2.8953831,
            6.02083,
            8.563837,
            10.9148445,
            14.125958,
            17.425943,
            19.70147,
        ],
        [
            3.100937,
            3.100937,
            6.584794,
            9.891179,
            12.366346,
            16.112404,
            19.496128,
            22.475035,
        ],
        [
            3.4451933,
            3.4451933,
            7.7654533,
            12.056869,
            15.51299,
            20.03222,
            24.362507,
            27.342232,
        ],
    ]
)

expected_sug_bradley_out = np.array(
    [
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ]
)

expected_sug_fuzzy_out = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
)


def test_compute_sat_sug_bradley_images():
    int_img = fuzzy_sat(test_image)

    int_img.compute_sat_sug()
    int_img.adaptive_thresh_bradley(3, 0.1)

    test_s = int_img.get_S()
    test_s_c = int_img.get_S_c()
    test_out = int_img.get_FTh()
    np.testing.assert_allclose(expected_s, test_s)
    np.testing.assert_allclose(expected_sug_s_c, test_s_c)
    np.testing.assert_allclose(test_out, expected_sug_bradley_out)


def test_compute_sat_sug_fuzzy_images():
    int_img = fuzzy_sat(test_image)

    int_img.compute_sat_sug()
    int_img.adaptive_thresh_fuzzy(3, 0.1)

    test_s = int_img.get_S()
    test_s_c = int_img.get_S_c()
    test_out = int_img.get_FTh()
    np.testing.assert_allclose(expected_s, test_s)
    np.testing.assert_allclose(expected_sug_s_c, test_s_c)
    np.testing.assert_allclose(test_out, expected_sug_fuzzy_out)
