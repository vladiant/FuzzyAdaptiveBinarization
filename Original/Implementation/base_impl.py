import argparse as ap

import numpy as np
import cv2


class fuzzy_sat:
    def __init__(self, image):
        shape = image.shape
        self.image = np.asarray(image, dtype=np.float32)
        self.height = shape[0]
        self.width = shape[1]
        self.S = np.zeros(shape, dtype=np.float32)  # Create an empty summed area table
        self.S_c = np.zeros(
            shape, dtype=np.float32
        )  # Create an empty summed area table
        self.out_img = np.zeros(shape, dtype=np.float32)
        self.th_mat = np.zeros(shape, dtype=np.float32)

    def compute_sat(self):  # Integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = (
                        self.image[row][col]
                        + self.S[row][col - 1]
                        + self.S[row - 1][col]
                        - self.S[row - 1][col - 1]
                    )
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                else:
                    self.S[row][col] = self.image[row][col]

    def compute_sat_sug(self):  # Sugeno integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = (
                        self.image[row][col]
                        + self.S[row][col - 1]
                        + self.S[row - 1][col]
                        - self.S[row - 1][col - 1]
                    )
                    if self.S[row][col - 1] >= self.S[row - 1][col]:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col - 1],
                                self.S[row][col],
                            ]
                        )
                    else:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col],
                            ]
                        )
                    self.S_c[row][col] = np.max(
                        [ov[1], min(ov[2], 0.75), min(ov[3], 0.5), min(ov[4], 0.25)]
                    )
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = np.max([ov[1], min(ov[2], 0.50)])
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                    ov = np.asarray([0, self.S[row][col - 1], self.S[row][col]])
                    self.S_c[row][col] = np.max([ov[1], min(ov[2], 0.50)])
                else:
                    self.S[row][col] = self.image[row][col]
                    self.S_c[row][col] = self.image[row][col]

    def compute_sat_cf12(self):  # CFval 1,2 integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = (
                        self.image[row][col]
                        + self.S[row][col - 1]
                        + self.S[row - 1][col]
                        - self.S[row - 1][col - 1]
                    )
                    if self.S[row][col - 1] >= self.S[row - 1][col]:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col - 1],
                                self.S[row][col],
                            ]
                        )
                    else:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col],
                            ]
                        )
                    self.S_c[row][col] = (
                        ov[1] + ov[2] * 0.75 + ov[3] * 0.5 + ov[4] * 0.25
                    )
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = ov[1] + ov[2] * 0.50
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                    ov = np.asarray([0, self.S[row][col - 1], self.S[row][col]])
                    self.S_c[row][col] = ov[1] + ov[2] * 0.50
                else:
                    self.S[row][col] = self.image[row][col]
                    self.S_c[row][col] = self.image[row][col]

    def compute_sat_cho(self):  # Choquet integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = (
                        self.image[row][col]
                        + self.S[row][col - 1]
                        + self.S[row - 1][col]
                        - self.S[row - 1][col - 1]
                    )
                    if self.S[row][col - 1] >= self.S[row - 1][col]:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col - 1],
                                self.S[row][col],
                            ]
                        )
                    else:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col],
                            ]
                        )
                    self.S_c[row][col] = (
                        (ov[1] - ov[0])
                        + (ov[2] - ov[1]) * 0.75
                        + (ov[3] - ov[2]) * 0.50
                        + (ov[4] - ov[3]) * 0.25
                    )
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) + (ov[2] - ov[1]) * 0.5
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                    ov = np.asarray([0, self.S[row][col - 1], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) + (ov[2] - ov[1]) * 0.5
                else:
                    self.S[row][col] = self.image[row][col]
                    self.S_c[row][col] = self.S[row][col]

    def compute_sat_ham(self):  # Hamacher integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = (
                        self.image[row][col]
                        + self.S[row][col - 1]
                        + self.S[row - 1][col]
                        - self.S[row - 1][col - 1]
                    )
                    if self.S[row][col - 1] >= self.S[row - 1][col]:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col - 1],
                                self.S[row][col],
                            ]
                        )
                    else:
                        ov = np.asarray(
                            [
                                0,
                                self.S[row - 1][col - 1],
                                self.S[row][col - 1],
                                self.S[row - 1][col],
                                self.S[row][col],
                            ]
                        )
                    self.S_c[row][col] = (
                        (ov[1] - ov[0]) / (ov[1] + 1 - (ov[1]))
                        + (ov[2] - ov[1]) / (ov[2] + 0.75 - (ov[2] * 0.75))
                        + (ov[3] - ov[2]) / (ov[3] + 0.50 - (ov[0] * 0.50))
                        + (ov[4] - ov[3]) / (ov[4] + 0.25 - (ov[4] * 0.25))
                    )
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) / (ov[1] + 1 - (ov[1])) + (
                        ov[2] - ov[1]
                    ) / (ov[2] + 0.5 - (ov[2] * 0.5))
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                    ov = np.asarray([0, self.S[row][col - 1], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) / (ov[1] + 1 - (ov[1])) + (
                        ov[2] - ov[1]
                    ) / (ov[2] + 0.5 - (ov[2] * 0.5))
                else:
                    self.S[row][col] = self.image[row][col]
                    self.S_c[row][col] = self.S[row][col]

    def adaptive_thresh_bradley(self, a1, T):  # adaptive thresholding
        w_n = min(self.height, self.width) / a1
        for col in range(self.width):
            for row in range(self.height):
                # SxS region
                y0 = int(max(row - w_n, 0))
                y1 = int(min(row + w_n, self.height - 1))
                x0 = int(max(col - w_n, 0))
                x1 = int(min(col + w_n, self.width - 1))

                count = (y1 - y0) * (x1 - x0)
                sum_ = self.S[y1, x1] - self.S[y0, x1] - self.S[y1, x0] + self.S[y0, x0]

                self.th_mat[row, col] = sum_ / count

                if self.image[row, col] * count < sum_ * (1.0 - T):
                    self.out_img[row, col] = 0
                else:
                    self.out_img[row, col] = 1

    def adaptive_thresh_fuzzy(self, a1, T):  # fuzzy adaptive thresholding
        w_n = min(self.height, self.width) / a1
        for col in range(self.width):
            for row in range(self.height):
                # SxS region
                y0 = int(max(row - w_n, 0))
                y1 = int(min(row + w_n, self.height - 1))
                x0 = int(max(col - w_n, 0))
                x1 = int(min(col + w_n, self.width - 1))

                count = (y1 - y0) * (x1 - x0)
                sum_ = (
                    self.S_c[y1, x1]
                    - self.S_c[y0, x1]
                    - self.S_c[y1, x0]
                    + self.S_c[y0, x0]
                )

                self.th_mat[row, col] = sum_ / count

                if self.image[row, col] * count < sum_ * (1.0 - T):
                    self.out_img[row, col] = 0
                else:
                    self.out_img[row, col] = 1

    def get_S(self):
        return self.S

    def get_S_c(self):
        return self.S_c

    def get_FTh(self):
        return self.out_img


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--image", help="image filename", required="True")
    args = vars(parser.parse_args())

    image_path = args["image"]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    int_img = fuzzy_sat(np.asarray(img))

    # int_img.compute_sat()
    # int_img.compute_sat_cf12()
    int_img.compute_sat_cho()
    # int_img.compute_sat_ham()

    # int_img.adaptive_thresh_bradley(16, 0.095)
    int_img.adaptive_thresh_fuzzy(16, 0.095)

    out_img = int_img.get_FTh()

    # Write intermediate images for comparison
    cv2.imwrite("test_s_c.tiff", int_img.get_S_c())
    cv2.imwrite("test_s.tiff", int_img.get_S())
    cv2.imwrite("test_fth.tiff", int_img.get_FTh())

    cv2.imshow("img", img)
    cv2.imshow("out", out_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
