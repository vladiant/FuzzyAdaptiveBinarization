#include <opencv2/core/mat.hpp>

#include "doctest/doctest.h"
#include "fuzzy_sat.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

namespace {

const cv::Mat1f expected_sug_s_c(
    {8, 8}, {
                0.02920038, 1.0016258, 1.4825196, 2.0967994,
                2.6659548,  3.2250404, 3.8745317, 4.4543095,

                0.92777306, 1.5218266, 2.2213287, 2.7313054,
                3.2135174,  3.9330854, 4.5139184, 5.1108055,

                1.382016,   2.1665874, 3.3273642, 4.3493834,
                5.1144004,  6.168821,  7.562674,  8.708733,

                1.9036568,  2.6951916, 4.5944977, 6.0736012,
                7.165868,   8.696261,  10.629386, 11.883804,

                2.5192761,  3.2486994, 5.7942376, 7.6626105,
                9.305782,   11.556599, 14.142697, 15.624799,

                2.9956303,  3.5866241, 6.6476254, 9.217769,
                11.574041,  14.754173, 17.966341, 20.320366,

                3.2558262,  3.9521112, 7.4611025, 10.755332,
                13.338937,  16.97396,  20.183336, 23.188137,

                3.6790159,  4.39311,   8.530791,  12.642855,
                16.09608,   20.605894, 24.841684, 27.819904,
            });

const cv::Mat1f expected_sug_bradley_out(
    {8, 8}, {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
             //  Python code results
         //  1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0
             1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

const cv::Mat1f expected_sug_fuzzy_out(
    {8, 8}, {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             //  Python code results
          // 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
             1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

}  // namespace

TEST_CASE("ComputeSatBradley [ham]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_ham();
  int_img.adaptive_thresh_bradley(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  compare_images(test_out, expected_sug_bradley_out);
}

TEST_CASE("ComputeSatFuzzy [ham]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat_ham();
  int_img.adaptive_thresh_fuzzy(3, 0.1);

  const auto test_s = int_img.get_S();
  const auto test_s_c = int_img.get_S_c();
  const auto test_out = int_img.get_FTh();

  compare_images(get_expected_s(), test_s);
  compare_images(expected_sug_s_c, test_s_c);
  compare_images(test_out, expected_sug_fuzzy_out);
}