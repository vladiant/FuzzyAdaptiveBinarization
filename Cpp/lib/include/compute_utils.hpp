#pragma once

#include <opencv2/core/mat.hpp>

// Integral image
void compute_integral(const cv::Mat1f& image, cv::Mat1f& S);

// Sugeno integral image
void compute_integral_sugeno(const cv::Mat1f& image, cv::Mat1f& S,
                             cv::Mat1f& S_c);

// CFval 1,2 integral image
void compute_integral_cf12(const cv::Mat1f& image, cv::Mat1f& S,
                           cv::Mat1f& S_c);

// Choquet integral image
void compute_integral_cho(const cv::Mat1f& image, cv::Mat1f& S, cv::Mat1f& S_c);

// Hamacher integral image
void compute_integral_ham(const cv::Mat1f& image, cv::Mat1f& S, cv::Mat1f& S_c);

// adaptive thresholding
void thresh_bradley(int a1, float T, const cv::Mat1f& image, const cv::Mat1f& S,
                    cv::Mat1f& out_img);

// fuzzy adaptive thresholding
void thresh_fuzzy(int a1, float T, const cv::Mat1f& image, const cv::Mat1f& S_c,
                  cv::Mat1f& out_img);
