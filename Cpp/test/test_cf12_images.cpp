#include <opencv2/core/mat.hpp>

#include "doctest/doctest.h"
#include "fuzzy_sat.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

namespace {

const cv::Mat1f expected_sug_s_c(
    {8, 8}, {
                2.92003788e-02, 5.30782580e-01, 1.82047689e00, 2.84783936e00,
                3.87273932e00,  4.72691345e00,  5.77084064e00, 6.67817450e00,

                4.63624597e-01, 1.65723348e00,  4.06134081e00, 6.31326389e00,
                8.04494476e00,  9.86549091e00,  1.22172823e01, 1.42246237e01,

                1.62578118e00,  3.83978796e00,  7.65938950e00, 1.10734730e01,
                1.32289829e01,  1.64554043e01,  2.05978012e01, 2.36766758e01,

                2.57507038e00,  6.11463690e00,  1.16438599e01, 1.56008720e01,
                1.88414803e01,  2.37882195e01,  2.89209499e01, 3.23408318e01,

                3.57010126e00,  8.27343845e00,  1.50204315e01, 2.00818005e01,
                2.52167568e01,  3.21241798e01,  3.83586311e01, 4.31874695e01,

                4.44585133e00,  9.87769890e00,  1.77141380e01, 2.45312309e01,
                3.12806854e01,  3.97972641e01,  4.75175323e01, 5.43660698e01,

                4.82353354e00,  1.09185925e01,  2.03686905e01, 2.89952507e01,
                3.70653458e01,  4.68412361e01,  5.53692169e01, 6.35736847e01,

                5.46203709e00,  1.25740910e01,  2.38233929e01, 3.41682549e01,
                4.38768654e01,  5.57144279e01,  6.57721481e01, 7.40754623e01,
            });

const cv::Mat1f expected_sug_bradley_out(
    {8, 8}, {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

const cv::Mat1f expected_sug_fuzzy_out(
    {8, 8}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

}  // namespace

TEST_CASE("ComputeSatBradley [cf12]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_cf12();
  int_img.adaptive_thresh_bradley(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  // compare_images(test_out, expected_sug_bradley_out);
}

TEST_CASE("ComputeSatFuzzy [cf12]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_cf12();
  int_img.adaptive_thresh_fuzzy(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  compare_images(test_out, expected_sug_fuzzy_out);
}