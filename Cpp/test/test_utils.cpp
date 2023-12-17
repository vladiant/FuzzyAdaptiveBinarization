#include "test_utils.hpp"

#include "doctest/doctest.h"

void compare_images(const cv::Mat1f& lhs, const cv::Mat1f& rhs) {
  REQUIRE(lhs.size() == rhs.size());

  auto it_rhs = rhs.begin();
  auto it_lhs = lhs.begin();
  size_t index = 0;
  for (; it_rhs != rhs.end(); ++it_rhs, ++it_lhs, index++) {
    CHECK_MESSAGE(doctest::Approx(*it_rhs) == *it_lhs, index);
  }
}