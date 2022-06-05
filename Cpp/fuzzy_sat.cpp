#include "fuzzy_sat.hpp"

#include "compute_utils.hpp"

FuzzySat::FuzzySat(const cv::Mat1f& input_image)
    : image{input_image},
      S{input_image.size(), 0.0f},
      S_c{input_image.size(), 0.0f},
      out_img{input_image.size(), 0.0f} {}

void FuzzySat::compute_sat() { compute_integral(image, S); }

void FuzzySat::compute_sat_sug() { compute_integral_sugeno(image, S, S_c); }

void FuzzySat::compute_sat_cf12() { compute_integral_cf12(image, S, S_c); }

void FuzzySat::compute_sat_cho() { compute_integral_cho(image, S, S_c); }

void FuzzySat::compute_sat_ham() { compute_integral_ham(image, S, S_c); }

void FuzzySat::adaptive_thresh_bradley(int a1, float T) {
  thresh_bradley(a1, T, image, S, out_img);
}

void FuzzySat::adaptive_thresh_fuzzy(int a1, float T) {
  thresh_fuzzy(a1, T, image, S_c, out_img);
}
