import argparse as ap

import numpy as np
import cv2

from skimage import filters  # threshold_otsu


def import_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_reverted = cv2.bitwise_not(img)
    norm_img = img_reverted / 255.0
    return norm_img


def compute_summed_area_table(image):
    # image is a 2-dimensional array containing ints or floats, with at least 1 element.
    height = len(image)
    width = len(image[0])
    new_image = [
        [0.0] * width for _ in range(height)
    ]  # Create an empty summed area table
    for row in range(0, height):
        for col in range(0, width):
            if (row > 0) and (col > 0):
                new_image[row][col] = (
                    image[row][col]
                    + new_image[row][col - 1]
                    + new_image[row - 1][col]
                    - new_image[row - 1][col - 1]
                )
            elif row > 0:
                new_image[row][col] = image[row][col] + new_image[row - 1][col]
            elif col > 0:
                new_image[row][col] = image[row][col] + new_image[row][col - 1]
            else:
                new_image[row][col] = image[row][col]
    return new_image


# Adaptive choquet
# OPT= 0 Hamacher
# OPT= 1 Discrete Choquet
# Opt= 2 Discrete Choquet with F1,F2 on the distributive property


def compute_choquet(choquet_order, fuzzy_mu, opt=0):
    C = 0
    if opt == 0:  # Choquet Hamacher
        for i in range(len(choquet_order) - 1):
            j = i + 1
            C = C + (choquet_order[j] * fuzzy_mu[i]) / (
                choquet_order[j] + fuzzy_mu[i] - (choquet_order[j] * fuzzy_mu[i])
            )
    if opt == 1:  # Choquet
        for i in range(len(choquet_order) - 1):
            j = i + 1
            C = C + ((choquet_order[j] - choquet_order[j - 1]) * fuzzy_mu[i])
    if opt == 2:  # Choquet F1 F2
        for i in range(len(choquet_order) - 1):
            j = i + 1
            C = C + (
                np.sqrt(choquet_order[j] * fuzzy_mu[i])
                - max((choquet_order[j] + fuzzy_mu[i] - 1), 0)
            )
    return C


def compute_sugeno(sugeno_order, fuzzy_mu):
    S = np.empty((1), float)

    for i in range(len(sugeno_order)):
        S = np.append(S, min(sugeno_order[i], fuzzy_mu[i]))
        # print(S)
        # print('sugeno: ' + str(choquet_order[j]) + " " + str(fuzzy_mu[i]) + " " + str(max(S)))
    return max(S)


## Integral Choquet and Sugeno image.
def adaptive_choquet_itegral(input_img, int_img, opt, log=False):

    h, w = input_img.shape
    th_mat = np.zeros(input_img.shape)
    choquet_mat = np.zeros(input_img.shape)
    sugeno_mat = np.zeros(input_img.shape)
    count_matrix = np.zeros(input_img.shape)

    for col in range(w):  # i
        for row in range(h):  # j
            # SxS region

            y0 = int(max(row - 1, 0))
            y1 = int(min(row, h - 1))
            x0 = int(max(col - 1, 0))
            x1 = int(min(col, w - 1))

            count = (y1 - y0) * (x1 - x0)
            count_matrix[row, col] = count
            choquet_order = -1
            sum_ = -1
            fuzzy_mu = -1
            if count == 0:
                if x0 == x1 and y0 == y1:
                    sum_ = int_img[y0, x0]
                    C_ = sum_
                    S_ = sum_
                if x1 == x0 and y0 != y1:
                    sum_ = (int_img[y1, x1] + int_img[y0, x1]) / 2
                    choquet_order = np.asarray([0, int_img[y0, x1], int_img[y1, x1]])
                    sugeno_order = np.asarray([int_img[y0, x1], int_img[y1, x1]])
                    fuzzy_mu = np.asarray([1, 0.5])
                    C_ = compute_choquet(choquet_order, fuzzy_mu, opt)
                    S_ = compute_sugeno(sugeno_order, fuzzy_mu)
                if y1 == y0 and x1 != x0:
                    sum_ = (int_img[y1, x1] + int_img[y1, x0]) / 2
                    choquet_order = np.asarray([0, int_img[y1, x0], int_img[y1, x1]])
                    sugeno_order = np.asarray([int_img[y1, x0], int_img[y1, x1]])
                    fuzzy_mu = np.asarray([1, 0.5])
                    C_ = compute_choquet(choquet_order, fuzzy_mu, opt)
                    S_ = compute_sugeno(sugeno_order, fuzzy_mu)
            else:
                sum_ = (
                    int_img[y1, x1]
                    - int_img[y0, x1]
                    - int_img[y1, x0]
                    + int_img[y0, x0]
                )
                if int_img[y0, x1] > int_img[y1, x0]:
                    choquet_order = np.asarray(
                        [
                            0,
                            int_img[y0, x0],
                            int_img[y1, x0],
                            int_img[y0, x1],
                            int_img[y1, x1],
                        ]
                    )
                    sugeno_order = np.asarray(
                        [
                            int_img[y0, x0],
                            int_img[y1, x0],
                            int_img[y0, x1],
                            int_img[y1, x1],
                        ]
                    )
                else:
                    choquet_order = np.asarray(
                        [
                            0,
                            int_img[y0, x0],
                            int_img[y0, x1],
                            int_img[y1, x0],
                            int_img[y1, x1],
                        ]
                    )
                    sugeno_order = np.asarray(
                        [
                            int_img[y0, x0],
                            int_img[y0, x1],
                            int_img[y1, x0],
                            int_img[y1, x1],
                        ]
                    )

                fuzzy_mu = np.asarray([1, 0.75, 0.50, 0.25])
                C_ = compute_choquet(choquet_order, fuzzy_mu, opt)
                S_ = compute_sugeno(sugeno_order, fuzzy_mu)

            th_mat[row, col] = sum_
            choquet_mat[row, col] = C_
            sugeno_mat[row, col] = S_

            if log:
                coords_window = np.zeros_like(input_img)

                # coords_window[x0:x1,y0:y1] = 1.0
                coords_window[y0, x0] = 0.2
                coords_window[y1, x0] = 0.4
                coords_window[y0, x1] = 0.6
                coords_window[y1, x1] = 0.8
                plot_it(coords_window)

                print("Search_region")
                print(
                    "x0:"
                    + str(x0)
                    + " x1:"
                    + str(x1)
                    + " y0:"
                    + str(y0)
                    + " y1:"
                    + str(y1)
                )
                print("Row:" + str(row) + " Col:" + str(col))
                print("Count: " + str(count))
                print("choquet fixed ordered and fuzzy mu")
                print(choquet_order)
                print(fuzzy_mu)
                print("choquet calculus")
                print(C_)
                print("sugeno calculus")
                print(S_)
                print("Input mat")
                print(input_img)
                print("Int img")
                print(int_img)
                print("I integral mat: ")
                print(th_mat)
                print("C_ choquet")
                print(choquet_mat)
                print("S_ sugeno")
                print(sugeno_mat)
                print("Count matrix")
                print(count_matrix)
                print("-------")

    return choquet_mat, sugeno_mat, count_matrix


# Novel choquet adaptive approach
def adaptive_thresh2(input_img, int_img, a1=4, a2=1, T=0, log=False):
    if T == 0:
        T = filters.threshold_otsu(input_img)
        T = T

    out_img_choquet = np.zeros_like(input_img)
    out_img_sugeno = np.zeros_like(input_img)
    choquet_mat = np.zeros_like(input_img)
    sugeno_mat = np.zeros_like(input_img)
    h, w = input_img.shape
    S = w / a1
    s2 = S / a2

    for col in range(w):
        for row in range(h):
            y0 = int(max(row - s2, 0))
            y1 = int(min(row + s2, h - 1))
            x0 = int(max(col - s2, 0))
            x1 = int(min(col + s2, w - 1))
            count = (y1 - y0) * (x1 - x0)
            sum_ = -1
            fuzzy_mu = -1
            if count == 0:
                if x0 == x1 and y0 == y1:
                    sum_ = int_img[y0, x0]
                    S_ = sum_
                if x1 == x0 and y0 != y1:
                    sum_ = int_img[y1, x1] - int_img[y0, x1]
                    sugeno_order = np.asarray([int_img[y0, x1], int_img[y1, x1]])
                    fuzzy_mu = np.asarray([1, 0.5])

                    S_ = compute_sugeno(sugeno_order, fuzzy_mu)
                if y1 == y0 and x1 != x0:
                    sum_ = int_img[y1, x1] - int_img[y1, x0]
                    sugeno_order = np.asarray([int_img[y1, x0], int_img[y1, x1]])
                    fuzzy_mu = np.asarray([1, 0.5])

                    S_ = compute_sugeno(sugeno_order, fuzzy_mu)
            else:
                sum_ = (
                    int_img[y1, x1]
                    - int_img[y0, x1]
                    - int_img[y1, x0]
                    + int_img[y0, x0]
                )
                if int_img[y0, x1] > int_img[y1, x0]:
                    sugeno_order = np.asarray(
                        [
                            int_img[y0, x0],
                            int_img[y1, x0],
                            int_img[y0, x1],
                            int_img[y1, x1],
                        ]
                    )
                else:
                    sugeno_order = np.asarray(
                        [
                            int_img[y0, x0],
                            int_img[y0, x1],
                            int_img[y1, x0],
                            int_img[y1, x1],
                        ]
                    )
                fuzzy_mu = np.asarray([1, 0.75, 0.50, 0.25])
                S_ = compute_sugeno(sugeno_order, fuzzy_mu)

            choquet_mat[row, col] = sum_ / count

            if input_img[row, col] * count < sum_ * (1.0 - T) / 1.0:
                out_img_choquet[row, col] = 0
            else:
                out_img_choquet[row, col] = 1

            sugeno_mat[row, col] = S_ / count
            # note is not only T
            if input_img[row, col] * count < S_ * (1.0 - T) / 1.0:
                out_img_sugeno[row, col] = 0
            else:
                out_img_sugeno[row, col] = 1

    return out_img_choquet, out_img_sugeno, choquet_mat, sugeno_mat, T


parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="image filename", required="True")
args = vars(parser.parse_args())

image_path = args["image"]
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

test_image = 1.0 - import_img(image_path)
S1 = np.asarray(compute_summed_area_table(test_image))

# Choquet Adaptive Thresh
choquet_mat, _, _ = adaptive_choquet_itegral(
    np.asarray(test_image), S1, 0, log=False  # t-norm version
)
out_img_adapt_choquet, _, _, _, T = adaptive_thresh2(
    np.asarray(test_image), np.asarray(choquet_mat), a1=16, a2=2, T=0.095, log=False
)  # con compute_summed_area table doesn't work.

# Choquet Adaptive Thresh
choquet_mat, _, _ = adaptive_choquet_itegral(
    np.asarray(test_image), S1, 1, log=False  # choquet int version
)
out_img_adapt_choquet2, _, _, _, T = adaptive_thresh2(
    np.asarray(test_image), np.asarray(choquet_mat), a1=16, a2=2, T=0.095, log=False
)  # con compute_summed_area table doesn't work.

cv2.imshow("img", img)
cv2.imshow("adapt_choquet", out_img_adapt_choquet)
cv2.imshow("adapt_choquet2", out_img_adapt_choquet2)
cv2.waitKey(0)

cv2.destroyAllWindows()
