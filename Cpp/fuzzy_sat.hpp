#pragma once

#include <opencv2/core/mat.hpp>

class FuzzySat {
 public:
  FuzzySat(const cv::Mat1f& input_image);

  // Integral image
  void compute_sat();

  // CFval 1,2 integral image
  void compute_sat_cf12();

  // Choquet integral image
  void compute_sat_cho();

  // Hamacher integral image
  void compute_sat_ham();

  // adaptive thresholding
  void adaptive_thresh_bradley(int a1, float T);

  // fuzzy adaptive thresholding
  void adaptive_thresh_fuzzy(int a1, float T);

  cv::Mat1f get_S() const { return S; }

  cv::Mat1f get_S_c() const { return S_c; }

  cv::Mat1f get_FTh() const { return out_img; }

 private:
  cv::Mat1f image;
  cv::Mat1f S;
  cv::Mat1f S_c;
  cv::Mat1f out_img;
};
