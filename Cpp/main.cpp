#include <iostream>
#include <opencv2/opencv.hpp>

#include "fuzzy_sat.hpp"

constexpr auto inputName = "Initial";
constexpr auto binarizedName = "Binarized";

void update(int, void* data) {
  if (!data) {
    std::cout << "Invalid pointer to callback data!\n";
    std::exit(EXIT_FAILURE);
  }

  const auto& input_img = *static_cast<cv::Mat*>(data);

  FuzzySat fuzzy_sat(input_img);
  // fuzzy_sat.compute_sat();
  // fuzzy_sat.compute_sat_cf12();
  fuzzy_sat.compute_sat_cho();
  // fuzzy_sat.compute_sat_ham();

  // fuzzy_sat.adaptive_thresh_bradley(16, 0.095);
  fuzzy_sat.adaptive_thresh_fuzzy(16, 0.095);

  // Write intermediate images for comparison
  cv::imwrite("test_s_c.tiff", fuzzy_sat.get_S_c());
  cv::imwrite("test_s.tiff", fuzzy_sat.get_S());
  cv::imwrite("test_fth.tiff", fuzzy_sat.get_FTh());

  // cv::imshow("test", fuzzy_sat.get_S_c() / 10000.0);
  // cv::imshow("test", fuzzy_sat.get_S() / 2055.0);

  const auto binarized = fuzzy_sat.get_FTh();

  cv::imshow(binarizedName, binarized);
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " image_name" << '\n';
    return EXIT_SUCCESS;
  }

  std::string image_name = argv[1];

  cv::Mat source = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
  if (source.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  // convert to float point data
  cv::Mat img;
  source.convertTo(img, CV_32F);
  img /= 255.0;

  cv::namedWindow(inputName);
  cv::imshow(inputName, img);

  cv::namedWindow(binarizedName);

  update(0, &img);

  while (true) {
    const char ch = cv::waitKey(0);
    if (ch == 27) {
      break;
    }
    if (ch == ' ') {
      update(0, &img);
    }
  }

  return EXIT_SUCCESS;
}
