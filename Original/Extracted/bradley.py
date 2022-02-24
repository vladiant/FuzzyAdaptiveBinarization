import argparse as ap

import numpy as np
import cv2

from skimage import filters  # threshold_otsu


def import_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_reverted = cv2.bitwise_not(img)
    norm_img = img_reverted / 255.0
    return norm_img


def get_int_img_m1(input_img):
    h, w = input_img.shape
    # integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row, col] = input_img[0 : row + 1, 0 : col + 1].sum()
    return int_img


## Classic Bradley Apprroach
def adaptive_thresh(input_img, int_img, a1=8, a2=2, T=0.15):
    out_img = np.zeros_like(input_img)
    h, w = input_img.shape
    S = w / a1
    s2 = S / a2
    th_mat = np.zeros(input_img.shape)
    for col in range(w):
        for row in range(h):
            # SxS region
            y0 = int(max(row - s2, 0))
            y1 = int(min(row + s2, h - 1))
            x0 = int(max(col - s2, 0))
            x1 = int(min(col + s2, w - 1))

            count = (y1 - y0) * (x1 - x0)
            sum_ = int_img[y1, x1] - int_img[y0, x1] - int_img[y1, x0] + int_img[y0, x0]

            th_mat[row, col] = sum_ / count

            if input_img[row, col] * count < sum_ * (1.0 - T) / 1.0:
                out_img[row, col] = 0
            else:
                out_img[row, col] = 1

    return np.asarray(out_img), th_mat


parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="image filename", required="True")
args = vars(parser.parse_args())

image_path = args["image"]
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
test_image = 1.0 - import_img(image_path)

T = filters.threshold_otsu(np.asarray(test_image))

# Bradley Adaptive Thresh
S1 = get_int_img_m1(test_image)
out_img_bradley, bradley_int_mat = adaptive_thresh(
    np.asarray(test_image), S1, a1=16, a2=2, T=T
)

cv2.imshow("img", test_image)
cv2.imshow("bradley", out_img_bradley)
cv2.waitKey(0)

cv2.destroyAllWindows()
