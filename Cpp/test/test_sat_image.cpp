#include <opencv2/core/mat.hpp>

#include "doctest/doctest.h"
#include "fuzzy_sat.hpp"
#include "test_data.hpp"
#include "test_utils.hpp"

TEST_CASE("ComputeSatSImage [sat]") {
  FuzzySat int_img(get_test_image());

  int_img.compute_sat();

  const auto test_s = int_img.get_S();

  compare_images(get_expected_s(), test_s);
}
