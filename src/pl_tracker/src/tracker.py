#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError


def callback(msg):
    global bridge
    global fgbg
    global fgbg_edge
    try:
        # Regular background subtraction
        cv_im = bridge.imgmsg_to_cv2(msg, "bgr8")
        fg_mask = fgbg.apply(cv_im)
        bg_mask = cv2.bitwise_not(fg_mask)
        fg_im = cv2.bitwise_and(cv_im,cv_im,mask = fg_mask)
        bg_im = cv2.bitwise_and(cv_im,cv_im,mask = bg_mask)

        # Edge-based background subtraction
        fg_mask_edge = fgbg_edge.apply(cv2.Canny(cv_im,100,200))
        bg_mask_edge = cv2.bitwise_not(fg_mask_edge)
        fg_im_edge = cv2.bitwise_and(cv_im,cv_im,mask = fg_mask_edge)
        bg_im_edge = cv2.bitwise_and(cv_im,cv_im,mask = bg_mask_edge)
        
    except CvBridgeError as e:
        print e

    # Publish streams
    try:
        global pub
        global pub_edge
        pub.publish(bridge.cv2_to_imgmsg(fg_im, "bgr8"))
        pub_edge.publish(bridge.cv2_to_imgmsg(fg_im_edge, "bgr8"))
    except CvBridgeError as e:
        print e

def imu_listener():
    # Init publishers
    global pub
    pub = rospy.Publisher('tracker/image_raw', Image, queue_size=50)
    global pub_edge
    pub_edge = rospy.Publisher('tracker/image_edge', Image, queue_size=50)

    # BG Subtractors and cv_bridge
    global fgbg
    fgbg = cv2.createBackgroundSubtractorMOG2()
    global fgbg_edge
    fgbg_edge = cv2.createBackgroundSubtractorMOG2()
    global bridge
    bridge = CvBridge()

    # Init ROS node
    rospy.init_node('pf_tracker')

    # Subscribe/Spin
    global sub
    sub = rospy.Subscriber("cam0/image_raw", Image, callback)
    rospy.spin()


if __name__ == '__main__':
    imu_listener()


