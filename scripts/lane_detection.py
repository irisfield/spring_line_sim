from numpy import ndarray
import cv2 as cv
import numpy as np
from utils import cols, rows
from spring_line_sim.cfg import BlobConfig


def compute_lines(self, config:BlobConfig, image:ndarray) -> ndarray:
        lines_mat = np.zeros_like(image)
        x = 0
        y = int(config.lines_top * rows(image))
        w = cols(image)
        h = int(rows(image) - config.lines_top * rows(image))

        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        canny_image = cv.Canny(gray_image, 200,255)

        image_cropped = canny_image[y:y+h, x:x+w]
        
        lines = cv.HoughLinesP(image_cropped,
                               rho=config.lines_rho,
                               theta=0.01745329251,
                               threshold=config.lines_thresh,
                               minLineLength=config.lines_min_len,
                               maxLineGap=config.lines_max_gap)
        if lines is not None:
            for l in lines:
                l = l[0] # (4,1) => (4,)
                diffx = l[0] - l[2]
                diffy = l[1] - l[3]

                slope = diffy / diffx

                if abs(slope) < config.lines_min_slope: continue

                diffx *= 5
                diffy *= 5

                l[0] -= diffx
                l[1] -= diffy
                l[2] += diffx
                l[3] += diffy

                cv.line(
                    lines_mat,
                    (l[0], int(l[1] + config.lines_top * rows(image))),
                    (l[2], int(l[3] + config.lines_top * rows(image))),
                    255, 5)

        return lines_mat



# def find_lanes(input_image: ndarray,
#                config: BlobConfig,
#                debug_image: ndarray=None) -> ndarray:
#     """
#     This algorithm uses light-on-dark contrast to find 
#     lane lines. If lanes do not have this property, another
#     lane-finding algorithm may be used instead
#     """

#     # Median blur
#     #image = cv.medianBlur(input_image, config.enhance_blur * 2 + 1)


#     ## Find edges using Laplacian and Sobel
#     # Canny could also be used here but the Laplacian/Sobel
#     # approach typically yeilds improved expiremental
#     # results for this case
#     # image = cv.Laplacian(image, -1, config.lapla_ksize * 2 + 1)
#     # image = cv.Sobel(image, -1, config.sobel_xorder,
#     #             config.sobel_yorder,
#     #             config.sobel_ksize * 2 + 1)
#     gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
#     image = cv.Canny(gray_image, 200,255)

#     # Dilate images

#     # dilation_size = (2 * config.blob_dilation_size + 1, 2 * config.blob_dilation_size + 1)
#     # dilation_anchor = (config.blob_dilation_size, config.blob_dilation_size)
#     # dilate_element = cv.getStructuringElement(cv.MORPH_RECT, dilation_size, dilation_anchor)
#     # image = cv.dilate(image, dilate_element)
#     image = compute_lines(image, config, debug_image=debug_image)

#     return image
