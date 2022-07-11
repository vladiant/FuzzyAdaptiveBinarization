import pytest
import numpy as np

from base_impl import fuzzy_sat
from test_sat_image import test_image, expected_s

expected_sug_s_c = np.array(
    [
        [
            0.02920038,
            1.0016258,
            1.4825196,
            2.0967994,
            2.6659548,
            3.2250404,
            3.8745317,
            4.4543095,
        ],
        [
            0.92777306,
            1.4324712,
            2.1110306,
            2.4574099,
            2.9094906,
            3.6845992,
            4.159089,
            4.7518573,
        ],
        [
            1.382016,
            2.0059574,
            3.3240674,
            4.1809077,
            4.9272404,
            6.0659122,
            7.371839,
            8.539515,
        ],
        [
            1.9036568,
            2.343718,
            4.5239234,
            6.0045147,
            7.0873666,
            8.647255,
            10.464166,
            11.730208,
        ],
        [
            2.5192761,
            2.871869,
            5.683778,
            7.635798,
            9.273182,
            11.517175,
            14.001486,
            15.494613,
        ],
        [
            2.9956303,
            3.1388447,
            6.4292808,
            9.12809,
            11.453734,
            14.680897,
            17.956177,
            20.316944,
        ],
        [
            3.2558262,
            3.5089648,
            7.2565327,
            10.731278,
            13.302855,
            16.947851,
            20.10742,
            23.144434,
        ],
        [
            3.6790159,
            3.9416203,
            8.278081,
            12.515095,
            15.949279,
            20.52442,
            24.829147,
            27.777206,
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
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
)


def test_compute_sat_ham_bradley_images():
    int_img = fuzzy_sat(test_image)

    int_img.compute_sat_ham()
    int_img.adaptive_thresh_bradley(3, 0.1)

    test_s = int_img.get_S()
    test_s_c = int_img.get_S_c()
    test_out = int_img.get_FTh()
    np.testing.assert_allclose(expected_s, test_s)
    np.testing.assert_allclose(expected_sug_s_c, test_s_c)
    np.testing.assert_allclose(test_out, expected_sug_bradley_out)


def test_compute_sat_ham_fuzzy_images():
    int_img = fuzzy_sat(test_image)

    int_img.compute_sat_ham()
    int_img.adaptive_thresh_fuzzy(3, 0.1)

    test_s = int_img.get_S()
    test_s_c = int_img.get_S_c()
    test_out = int_img.get_FTh()
    np.testing.assert_allclose(expected_s, test_s)
    np.testing.assert_allclose(expected_sug_s_c, test_s_c)
    np.testing.assert_allclose(test_out, expected_sug_fuzzy_out)
