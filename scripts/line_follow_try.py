#!/usr/bin/env python3
# https://www.youtube.com/watch?v=AbqErp4ZGgU
# https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
# https://towardsdatascience.com/finding-driving-lane-line-live-with-opencv-f17c266f15db


import cv2
import math
import rospy
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server

from typing import Dict, Tuple, List
from numpy import ndarray
from vec import Vec
from lane_centering import center_lane
from lane_detection import compute_lines
from utils import rows, cols
from spring_line_sim.cfg import BlobConfig

# global variables
yaw_rate = Float32()
debug_publishers: Dict[str, rospy.Publisher] = {}
cvbridge = CvBridge()

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

################### callback ###################
def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    return config

def image_callback(camera_image):

    try:
        # convert camera_image into an opencv-compatible image
        cv_image = cvbridge.imgmsg_to_cv2(camera_image, "bgr8")
    except CvBridgeError:
        print(CvBridgeError)

    # get the dimensions of the image
    width = cv_image.shape[1]
    height = cv_image.shape[0]

    debug_image = cv_image.copy()

    # Find the lanes in the image
    lanes_image = compute_lines(cv_image, RC, debug_image)
    debug_publish('lanes_image', lanes_image)

    adjust = blob_adjust(lanes_image, debug_image=debug_image)
    debug_publish('debug_final', debug_image)
    # Convert the output to a Twist message
    make_twist(adjust)

    cv2.imshow("CV Image", cv_image)
    cv2.imshow("Hough Lines", lanes_image)
    cv2.imshow("Springs", debug_image)
    cv2.waitKey(3)
    #rate.sleep()


################### algorithms ###################
def blob_adjust(image: ndarray, debug_image:ndarray=None) -> float:
        """
        Essentially a direct port from the original c++ algorithm
        """

        #Dilate images

        dilation_size = (2 * RC.blob_dilation_size + 1, 2 * RC.blob_dilation_size + 1)
        dilation_anchor = (RC.blob_dilation_size, RC.blob_dilation_size)
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_size, dilation_anchor)
        image = cv2.dilate(image, dilate_element)


        if RC.blob_median_blur_size > 0:
            image = cv2.medianBlur(image, RC.blob_median_blur_size * 2 + 1);

        points = []
        theta = 0
        while theta <= math.pi:

            p = Point(RC.blob_x, RC.blob_y)
            diffx = math.cos(theta) * 0.01
            diffy = -1 * math.sin(theta) * 0.01

            # NOTE: may need to switch x/y here
            while image[int(p.y * rows(image)), int(p.x * cols(image))] < RC.blob_num_points:
                p.x += diffx
                p.y += diffy

                top_y = 1 - RC.blob_max_p_y
                if p.y > 1 or p.y < top_y or p.x > 1 or p.x < 0:
                    if p.x > 1: p.x = 1
                    if p.x < 0: p.x = 0
                    if p.y > 1: p.y = 1
                    if p.y < top_y: p.y = top_y
                    break

            if debug_image is not None:
                cv.circle(debug_image, 
                          (int(p.x * cols(debug_image)), int(p.y * rows(debug_image))),
                          5, (255, 0, 0), -1)

            points.append(p)

            theta += math.pi / RC.blob_num_points

        center_p = Point(RC.blob_x, RC.blob_y)
        center_a = Point(0, 0)

        for p in points:
            diffx = center_p.x - p.x
            diffy = center_p.y - p.y

            center_a.x += p.x
            center_a.y += p.y

            length = math.sqrt(diffx * diffx + diffy * diffy);

            if length < .01: continue

            diffx /= length
            diffy /= length

            spring_force = -1 * RC.blob_coeff * (length - RC.blob_len)

            diffx *= spring_force
            diffy *= spring_force

            center_p.x = diffx
            center_p.y = diffy

        center_a.x /= len(points)
        center_a.y /= len(points)

        center_p.x = center_a.x + center_p.x
        center_p.y = center_a.y + center_p.y

        if debug_image is not None:
            cv2.circle(debug_image, 
                      (int(center_p.x * cols(debug_image)), int(center_p.y * rows(debug_image))),
                      5, (0, 0, 255), -1)
            cv2.circle(debug_image,
                      (int(RC.blob_x * cols(debug_image)), int(RC.blob_y * rows(debug_image))),
                      5, (0, 255, 0), -1)

        return center_p.x - RC.blob_x

def make_twist(turn):
    angular_z = - RC.blob_mult * turn
    yaw_rate.data = angular_z
    yaw_rate_pub.publish(yaw_rate)
    return

def debug_publish(name, image: ndarray):
        name = f'/lane_follow_blob_debug/{name}'
        if name not in debug_publishers:
            debug_publishers[name] = rospy.Publisher(name, Image, queue_size=2)
        debug_publishers[name].publish(cvbridge.cv2_to_imgmsg(image))

################### main ###################

if __name__ == "__main__":

    rospy.init_node("follow_line", anonymous=True)

    rospy.Subscriber("/camera_view", Image, image_callback)
    dynamic_reconfigure_server = Server(BlobConfig, dynamic_reconfigure_callback)

    #rate = rospy.Rate(10)
    yaw_rate_pub = rospy.Publisher("yaw_rate", Float32, queue_size=1)

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
