import cv2
import numpy as np
from scipy import signal

def abs_sobel(img_ch, orient='x', sobel_kernel=3):

    """

    Applies the sobel operation on a gray scale image.



    :param img_ch:

    :param orient: 'x' or 'y'

    :param sobel_kernel: an uneven integer

    :return:

    """

    if orient == 'x':

        axis = (1, 0)

    elif orient == 'y':

        axis = (0, 1)

    else:

        raise ValueError('orient has to be "x" or "y" not "%s"' % orient)



    sobel = cv2.Sobel(img_ch, -1, *axis, ksize=sobel_kernel)

    abs_s = np.absolute(sobel)



    return abs_s





def gradient_magnitude(sobel_x, sobel_y):

    """

    Calculates the magnitude of the gradient.

    :param sobel_x:

    :param sobel_y:

    :return:

    """

    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return abs_grad_mag.astype(np.uint16)





def gradient_direction(sobel_x, sobel_y):

    """

    Calculates the direction of the gradient. NaN values cause by zero division will be replaced

    by the maximum value (np.pi / 2).

    :param sobel_x:

    :param sobel_y:

    :return:

    """

    with np.errstate(divide='ignore', invalid='ignore'):

        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))

        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2



    return abs_grad_dir.astype(np.float32)





def gaussian_blur(img, kernel_size):

    """

    Applies a Gaussian Noise kernel

    :param img:

    :param kernel_size:

    :return:

    """

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)





def extract_yellow(img):

    """

    Generates an image mask selecting yellow pixels.

    :param img: image with pixels in range 0-255

    :return: Yellow 255 not yellow 0

    """

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))



    return mask





def extract_dark(img):

    """

    Generates an image mask selecting dark pixels.

    :param img: image with pixels in range 0-255

    :return: Dark 255 not dark 0

    """

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, (0, 0, 0.), (255, 153, 128))

    return mask





def extract_highlights(img, p=99.9):

    """

    Generates an image mask selecting highlights.

    :param p: percentile for highlight selection. default=99.9

    :param img: image with pixels in range 0-255

    :return: Highlight 255 not highlight 0

    """

    p = int(np.percentile(img, p) - 30)

    mask = cv2.inRange(img, p, 255)

    return mask





def binary_noise_reduction(img, thresh):

    """

    Reduces noise of a binary image by applying a filter which counts neighbours with a value

    and only keeping those which are above the threshold.

    :param img: binary image (0 or 1)

    :param thresh: min number of neighbours with value

    :return:

    """

    k = np.array([[1, 1, 1],

                  [1, 0, 1],

                  [1, 1, 1]])

    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)

    img[nb_neighbours < thresh] = 0

    return img





def generate_lane_mask(img, v_cutoff=0):

    """

    Generates a binary mask selecting the lane lines of an street scene image.

    :param img: RGB color image

    :param v_cutoff: vertical cutoff to limit the search area

    :return: binary mask

    """

    window = img[v_cutoff:, :, :]

    yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)

    yuv = 255 - yuv

    hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)

    chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)

    gray = np.mean(chs, 2)



    s_x = abs_sobel(gray, orient='x', sobel_kernel=3)

    s_y = abs_sobel(gray, orient='y', sobel_kernel=3)



    grad_dir = gradient_direction(s_x, s_y)

    grad_mag = gradient_magnitude(s_x, s_y)



    ylw = extract_yellow(window)

    highlights = extract_highlights(window[:, :, 0])



    mask = np.zeros(img.shape[:-1], dtype=np.uint8)



    mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &

                        (s_y >= 25) & (s_y <= 255)) |

                       ((grad_mag >= 30) & (grad_mag <= 512) &

                        (grad_dir >= 0.2) & (grad_dir <= 1.)) |

                       (ylw == 255) |

                       (highlights == 255)] = 1



    mask = binary_noise_reduction(mask, 4)



    return mask