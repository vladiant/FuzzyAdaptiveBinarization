#include <opencv2/core/mat.hpp>

#include "doctest/doctest.h"
#include "fuzzy_sat.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

namespace {

const cv::Mat1f expected_sug_s_c(
    {8, 8}, {
                2.92003788e-02, 5.16182423e-01, 1.31889462e00, 2.03052688e00,
                2.65952492e00,  3.28060317e00,  3.93654799e00, 4.57591915e00,
                4.49024409e-01, 9.50117886e-01, 2.01684022e00, 2.99150229e00,
                3.68515253e00,  4.60000277e00,  5.66561651e00, 6.59820604e00,
                1.19135690e00,  1.95640421e00,  3.58387804e00, 4.89784336e00,
                5.84351206e00,  7.36718082e00,  9.00475597e00, 1.02497864e01,
                1.81813753e00,  3.03212190e00,  5.17126799e00, 6.70944786e00,
                8.20400238e00,  1.04304924e01,  1.24183998e01, 1.39310741e01,
                2.50889635e00,  3.99679780e00,  6.59094954e00, 8.65751362e00,
                1.09760818e01,  1.39109879e01,  1.63928394e01, 1.85553837e01,
                2.99815989e00,  4.65048599e00,  7.76516008e00, 1.04340515e01,
                1.33798885e01,  1.67901077e01,  1.97746429e01, 2.27139683e01,
                3.27306509e00,  5.22409439e00,  9.07457352e00, 1.24568462e01,
                1.60059910e01,  2.00008144e01,  2.34189758e01, 2.68191280e01,
                3.73944068e00,  6.03198814e00,  1.05432768e01, 1.45137711e01,
                1.87250366e01,  2.34311943e01,  2.71681232e01, 3.07480469e01,
            });

const cv::Mat1f expected_sug_bradley_out(
    {8, 8}, {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

const cv::Mat1f expected_sug_fuzzy_out(
    {8, 8}, {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});

}  // namespace

TEST_CASE("ComputeSatBradley [cho]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_cho();
  int_img.adaptive_thresh_bradley(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  // compare_images(test_out, expected_sug_bradley_out);
}

TEST_CASE("ComputeSatFuzzy [cho]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_cho();
  int_img.adaptive_thresh_fuzzy(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  // compare_images(test_out, expected_sug_fuzzy_out);
}