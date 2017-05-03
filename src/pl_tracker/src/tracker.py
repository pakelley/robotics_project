#!/usr/bin/env python

#AUTHORS: WOLFE MAGNUS, PATRICK KELLEY

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

#downscaling of an image by 50% in both dimensions (gaussian)
def downscale(opencv_image):
    return cv2.pyrDown(opencv_image)

#optical flow computation
def opflow(next_image):
    global prev_cv_im
    next_cv_im = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
    if prev_cv_im[0][0] == -1:
        #of_val = cv2.optflow.calcOpticalFlowSF(next_image, next_image, 3, 2, 10)
        of_val = cv2.calcOpticalFlowFarneback(next_cv_im,next_cv_im,None,.5,3,15,3,7,1.5,0)
    else:
        #of_val = cv2.optflow.calcOpticalFlowSF(prev_cv_im, next_image, 3, 2, 10)
        of_val = cv2.calcOpticalFlowFarneback(prev_cv_im,next_cv_im,None,.5,3,15,3,7,1.5,0)
    prev_cv_im = next_cv_im
    return of_val

#conversion from optical flow rawdata to image
def of_image_conv(opflow_raw):
    h,w = opflow_raw.shape[:2]
    fx,fy = opflow_raw[:,:,0], opflow_raw[:,:,1]
    ang = np.arctan2(opflow_raw[:,:,0],opflow_raw[:,:,1])+np.pi
    v = np.sqrt(fx*fx + fy*fy)
    hsv = np.zeros((h,w,3),np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,2] = cv2.normalize(v,None,0,255,cv2.NORM_MINMAX)

def callback(msg):
    global bridge
    global fgbg
    global fgbg_edge
    global prev_cv_im
    try:
        # Image preprocessing - ROS -> OPENCV bridging and downscaling
        cv_im = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_im_down = downscale(cv_im)

        #PIPELINE_01: Optical Flow Image
        #of_im is visual representation of opflow
        #of_raw is the raw optical flow vectors
        of_raw = opflow(cv_im_down)
        of_im = of_image_conv(of_raw)
        
        #PIPELINE_02: Edge Image of the Optical Flow
        of_im_edge = (cv2.Canny(of_im,50,100))
        of_im_edge_col = cv2.cvtColor(of_im_edge, cv2.COLOR_GRAY2BGR)
        
        #PIPELINE_03: Mask Optical Flow Image
        # of_mask_im = 

        #PIPELINE_04: Optical Flow Mean Calculation
        mean_opflow = np.mean(of_im)

        #PIPELINE_05: Optical Flow Mean Subtraction
        #of_sub_im = of_mask_im - mean_opflow

        #PIPELINE_06: Threshold Mean-Subtracted Image
        # thresh_sub_im = 

        #PIPELINE_07: Re-Sample Particles
        # ptcl_im =
        # ptcl_overlay_im = 

        # Regular background subtraction
        fg_mask = fgbg.apply(cv_im_down)
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


