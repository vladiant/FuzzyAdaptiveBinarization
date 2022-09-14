#include <opencv2/core/mat.hpp>

#include "doctest/doctest.h"
#include "fuzzy_sat.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

namespace {

const cv::Mat1f expected_sug_s_c(
    {8, 8}, {
                0.02920038, 0.5,        1.0031644, 1.634625,
                2.4264288,  2.892621,   3.6685853, 4.2045107,

                0.5,        0.75,       1.0031644, 1.634625,
                2.4264288,  2.892621,   3.6685853, 4.2045107,

                0.86884844, 0.86884844, 1.8992584, 3.5303135,
                4.374642,   5.046919,   6.7918863, 7.9974837,

                1.5138655,  1.5138655,  3.5436444, 5.362296,
                6.3241215,  7.628366,   10.001552, 11.228103,

                2.1224096,  2.1224096,  4.9485683, 6.830563,
                8.320811,   10.542711,  13.549339, 14.894605,

                2.8953831,  2.8953831,  6.02083,   8.563837,
                10.9148445, 14.125958,  17.425943, 19.70147,

                3.100937,   3.100937,   6.584794,  9.891179,
                12.366346,  16.112404,  19.496128, 22.475035,

                3.4451933,  3.4451933,  7.7654533, 12.056869,
                15.51299,   20.03222,   24.362507, 27.342232,
            });

const cv::Mat1f expected_sug_bradley_out(
    {8, 8}, {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

const cv::Mat1f expected_sug_fuzzy_out(
    {8, 8}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});

}  // namespace

TEST_CASE("ComputeSatBradley [sug]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_sug();
  int_img.adaptive_thresh_bradley(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  // compare_images(test_out, expected_sug_bradley_out);
}

TEST_CASE("ComputeSatFuzzy [sug]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_sug();
  int_img.adaptive_thresh_fuzzy(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  compare_images(test_out, expected_sug_fuzzy_out);
}