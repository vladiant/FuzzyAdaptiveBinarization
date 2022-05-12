#include "fuzzy_sat.hpp"

FuzzySat::FuzzySat(const cv::Mat1f& input_image)
    : image{input_image},
      S{input_image.size(), 0.0f},
      S_c{input_image.size(), 0.0f},
      out_img{input_image.size(), 0.0f} {}

void FuzzySat::compute_sat() {
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      if (row > 0 && col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1) +
            S.at<float>(row - 1, col) - S.at<float>(row - 1, col - 1);
      } else if (row > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row - 1, col);
      } else if (col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1);
      } else {
        S.at<float>(row, col) = image.at<float>(row, col);
      }
    }
  }
}

void FuzzySat::compute_sat_cf12() {
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      if (row > 0 && col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1) +
            S.at<float>(row - 1, col) - S.at<float>(row - 1, col - 1);
        float ov1, ov2, ov3, ov4;
        if (S.at<float>(row, col - 1) >= S.at<float>(row - 1, col)) {
          ov1 = S.at<float>(row - 1, col - 1);
          ov2 = S.at<float>(row - 1, col);
          ov3 = S.at<float>(row, col - 1);
          ov4 = S.at<float>(row, col);
        } else {
          ov1 = S.at<float>(row - 1, col - 1);
          ov2 = S.at<float>(row, col - 1);
          ov3 = S.at<float>(row - 1, col);
          ov4 = S.at<float>(row, col);
        }
        S_c.at<float>(row, col) = ov1 + 0.75 * ov2 + 0.5 * ov3 + 0.25 * ov4;
      } else if (row > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row - 1, col);
        const float ov1 = S.at<float>(row - 1, col);
        const float ov2 = S.at<float>(row, col);
        S_c.at<float>(row, col) = ov1 + 0.5 * ov2;
      } else if (col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1);
        const float ov1 = S.at<float>(row, col - 1);
        const float ov2 = S.at<float>(row, col);
        S_c.at<float>(row, col) = ov1 + 0.5 * ov2;
      } else {
        S.at<float>(row, col) = image.at<float>(row, col);
        S_c.at<float>(row, col) = image.at<float>(row, col);
      }
    }
  }
}

void FuzzySat::compute_sat_cho() {
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      if (row > 0 && col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1) +
            S.at<float>(row - 1, col) - S.at<float>(row - 1, col - 1);
        float ov1, ov2, ov3, ov4;
        if (S.at<float>(row, col - 1) >= S.at<float>(row - 1, col)) {
          ov1 = S.at<float>(row - 1, col - 1);
          ov2 = S.at<float>(row - 1, col);
          ov3 = S.at<float>(row, col - 1);
          ov4 = S.at<float>(row, col);
        } else {
          ov1 = S.at<float>(row - 1, col - 1);
          ov2 = S.at<float>(row, col - 1);
          ov3 = S.at<float>(row - 1, col);
          ov4 = S.at<float>(row, col);
        }
        S_c.at<float>(row, col) =
            ov1 + 0.75 * (ov2 - ov1) + 0.5 * (ov3 - ov2) + 0.25 * (ov4 - ov3);
      } else if (row > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row - 1, col);
        const float ov1 = S.at<float>(row - 1, col);
        const float ov2 = S.at<float>(row, col);
        S_c.at<float>(row, col) = ov1 + 0.5 * (ov2 - ov1);
      } else if (col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1);
        const float ov1 = S.at<float>(row, col - 1);
        const float ov2 = S.at<float>(row, col);
        S_c.at<float>(row, col) = ov1 + 0.5 * (ov2 - ov1);
      } else {
        S.at<float>(row, col) = image.at<float>(row, col);
        S_c.at<float>(row, col) = image.at<float>(row, col);
      }
    }
  }
}

void FuzzySat::compute_sat_ham() {
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      if (row > 0 && col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1) +
            S.at<float>(row - 1, col) - S.at<float>(row - 1, col - 1);
        float ov1, ov2, ov3, ov4;
        if (S.at<float>(row, col - 1) >= S.at<float>(row - 1, col)) {
          ov1 = S.at<float>(row - 1, col - 1);
          ov2 = S.at<float>(row - 1, col);
          ov3 = S.at<float>(row, col - 1);
          ov4 = S.at<float>(row, col);
        } else {
          ov1 = S.at<float>(row - 1, col - 1);
          ov2 = S.at<float>(row, col - 1);
          ov3 = S.at<float>(row - 1, col);
          ov4 = S.at<float>(row, col);
        }
        S_c.at<float>(row, col) = ov1 +
                                  (ov2 * 0.75) / (ov2 + 0.75 - (ov2 * 0.75)) +
                                  (ov3 * 0.50) / (ov3 + 0.50 - (ov3 * 0.50)) +
                                  (ov4 * 0.25) / (ov4 + 0.25 - (ov4 * 0.25));
      } else if (row > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row - 1, col);
        const float ov1 = S.at<float>(row - 1, col);
        const float ov2 = S.at<float>(row, col);
        S_c.at<float>(row, col) = ov1 + (ov2 - ov1) / (ov2 + 0.5 - (ov2 * 0.5));
      } else if (col > 0) {
        S.at<float>(row, col) =
            image.at<float>(row, col) + S.at<float>(row, col - 1);
        const float ov1 = S.at<float>(row, col - 1);
        const float ov2 = S.at<float>(row, col);
        S_c.at<float>(row, col) = ov1 + (ov2 - ov1) / (ov2 + 0.5 - (ov2 * 0.5));
      } else {
        S.at<float>(row, col) = image.at<float>(row, col);
        S_c.at<float>(row, col) = image.at<float>(row, col);
      }
    }
  }
}

void FuzzySat::adaptive_thresh_bradley(int a1, float T) {
  const int w_n = std::min(image.rows, image.cols) / a1;
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      // SxS region
      const int y0 = std::max(row - w_n, 0);
      const int y1 = std::min(row + w_n, image.rows - 1);
      const int x0 = std::max(col - w_n, 0);
      const int x1 = std::min(col + w_n, image.cols - 1);

      const int count = (y1 - y0) * (x1 - x0);
      const float sum_ = S.at<float>(y1, x1) - S.at<float>(y0, x1) -
                         S.at<float>(y1, x0) + S.at<float>(y0, x0);

      out_img.at<float>(row, col) =
          image.at<float>(row, col) * count >= sum_ * (1.0 - T);
    }
  }
}

void FuzzySat::adaptive_thresh_fuzzy(int a1, float T) {
  const int w_n = std::min(image.rows, image.cols) / a1;
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      // SxS region
      const int y0 = std::max(row - w_n, 0);
      const int y1 = std::min(row + w_n, image.rows - 1);
      const int x0 = std::max(col - w_n, 0);
      const int x1 = std::min(col + w_n, image.cols - 1);

      const int count = (y1 - y0) * (x1 - x0);
      const float sum_ = S_c.at<float>(y1, x1) - S_c.at<float>(y0, x1) -
                         S_c.at<float>(y1, x0) + S_c.at<float>(y0, x0);

      out_img.at<float>(row, col) =
          image.at<float>(row, col) * count >= sum_ * (1.0 - T);
    }
  }
}
