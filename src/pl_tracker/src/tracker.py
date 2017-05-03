#!/usr/bin/env python

#AUTHORS: WOLFE MAGNUS, PATRICK KELLEY

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class Tracker:
    def __init__(self):
        # Init publishers
        self.pub_raw_im       = rospy.Publisher('tracker/raw_im',       Image, queue_size=50) #   0: Original
        self.pub_edge         = rospy.Publisher('tracker/edge',         Image, queue_size=50) #   /: Edge
        self.pub_opflow       = rospy.Publisher('tracker/opflow',       Image, queue_size=50) #   1: OpFlow
        self.pub_opflow_edge  = rospy.Publisher('tracker/opflow_edge',  Image, queue_size=50) #   2: OpFlow Edge
        # self.pub_opflow_mask  = rospy.Publisher('tracker/opflow_mask',  Image, queue_size=50) #   3: Masked OpFlow
        # self.pub_opflow_sub   = rospy.Publisher('tracker/opflow_sub',   Image, queue_size=50) #   5: MeanSub
        # self.pub_thresh_sub   = rospy.Publisher('tracker/thresh_sub',   Image, queue_size=50) #   6: Threshed MeanSub
        # self.pub_ptcls        = rospy.Publisher('tracker/ptcls',        Image, queue_size=50) # 7.a: Particles
        # self.pub_ptcl_overlay = rospy.Publisher('tracker/ptcl_overlay', Image, queue_size=50) # 7.b: Particle Overlay
    
        # Init BG Subtractors and cv_bridge
        self.fgbg      = cv2.createBackgroundSubtractorMOG2()
        self.fgbg_edge = cv2.createBackgroundSubtractorMOG2()
        self.bridge    = CvBridge()
    
        # Init opflow
        self.prev_cv_im = np.array([[-1]])
    
    def imu_listener(self):
        # Init ROS node
        rospy.init_node('pf_tracker')

        # Subscribe/Spin
        self.sub = rospy.Subscriber("cam0/image_raw", Image, self.track)
        rospy.spin()

    def track(self, msg):
        try:
            # Image preprocessing - ROS -> OPENCV bridging and downscaling
            cv_im      = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_im_down = self.downscale(cv_im)

            #PIPELINE_01: Optical Flow Image
            #of_im is visual representation of opflow
            #of_raw is the raw optical flow vectors
            of_raw = self.opflow(cv_im_down)
            of_im  = self.of_image_conv(of_raw)

            #PIPELINE_02: Edge Image of the Optical Flow
            of_edge_im     = (cv2.Canny(of_im,50,100))
            of_edge_im_col = cv2.cvtColor(of_edge_im, cv2.COLOR_GRAY2BGR)
        
            #PIPELINE_03: Mask Optical Flow Image
            # Regular background subtraction
            fg_im, bg_im, fg_mask, bg_mask = self.bg_subtract(cv_im_down, self.fgbg)

            # Edge-based background subtraction
            # (fg_edge_im, bg_edge_im, fg_edge_mask, bg_edge_mask) = self.bg_subtract(cv2.Canny(cv_im,100,200), self.fgbg_edge)
            (fg_edge_im, bg_edge_im, fg_edge_mask, bg_edge_mask) = self.bg_subtract(cv2.Canny(cv_im_down,100,200), self.fgbg_edge)
             
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

        except CvBridgeError as e:
            print e

        try:
            self.publish(fg_im, fg_edge_im, of_im, of_edge_im_col) # , of_mask_im, of_sub_im, thresh_sub_im, ptcl_im, ptcl_overlay_im
        except CvBridgeError as e:
            print e
            

    def bg_subtract(self, im, bg_subtractor):
        fg_mask = bg_subtractor.apply(im)
        bg_mask = cv2.bitwise_not(fg_mask)
        fg_im   = cv2.bitwise_and(im,im,mask = fg_mask)
        bg_im   = cv2.bitwise_and(im,im,mask = bg_mask)
        return (fg_im, bg_im, fg_mask, bg_mask)

    #downscaling of an image by 50% in both dimensions (gaussian)
    def downscale(self, opencv_image):
        return cv2.pyrDown(opencv_image)

    #optical flow computation
    def opflow(self, next_image):
        next_cv_im = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
        if self.prev_cv_im[0][0] == -1:
            #of_val = cv2.optflow.calcOpticalFlowSF(next_image, next_image, 3, 2, 10)
            of_val = cv2.calcOpticalFlowFarneback(next_cv_im,next_cv_im,None,.5,3,15,3,7,1.5,0)
        else:
            #of_val = cv2.optflow.calcOpticalFlowSF(self.prev_cv_im, next_image, 3, 2, 10)
            of_val = cv2.calcOpticalFlowFarneback(self.prev_cv_im,next_cv_im,None,.5,3,15,3,7,1.5,0)
        self.prev_cv_im = next_cv_im
        return of_val

    #conversion from optical flow rawdata to image
    def of_image_conv(self, opflow_raw):
        h,w = opflow_raw.shape[:2]
        fx,fy = opflow_raw[:,:,0], opflow_raw[:,:,1]
        ang = np.arctan2(opflow_raw[:,:,0],opflow_raw[:,:,1])+np.pi
        v = np.sqrt(fx*fx + fy*fy)
        hsv = np.zeros((h,w,3),np.uint8)
        hsv[...,1] = 255
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,2] = cv2.normalize(v,None,0,255,cv2.NORM_MINMAX)
        return hsv

    def publish(self, fg_im, fg_edge_im, of_im, of_edge_im_col):
        # Publish streams
        
        fg_edge_col = cv2.cvtColor(fg_edge_im,cv2.COLOR_GRAY2BGR)
            
        self.pub_raw_im.publish(      self.bridge.cv2_to_imgmsg(fg_im,           "bgr8"))
        self.pub_edge.publish(        self.bridge.cv2_to_imgmsg(fg_edge_col,     "bgr8"))
        self.pub_opflow.publish(      self.bridge.cv2_to_imgmsg(of_im,           "bgr8"))
        self.pub_opflow_edge.publish( self.bridge.cv2_to_imgmsg(of_edge_im_col,  "bgr8"))
        # self.pub_opflow_sub.publish(  self.bridge.cv2_to_imgmsg(of_mask_im,      "bgr8")) #TODO
        # self.pub_opflow_sub.publish(  self.bridge.cv2_to_imgmsg(of_sub_im,       "bgr8"))
        # self.pub_thresh_sub.publish(  self.bridge.cv2_to_imgmsg(thresh_sub_im,   "bgr8")) #TODO
        # self.pub_ptcls.publish(       self.bridge.cv2_to_imgmsg(ptcl_im,         "bgr8")) #TODO
        # self.pub_ptcl_overlay.publish(self.bridge.cv2_to_imgmsg(ptcl_overlay_im, "bgr8")) #TODO
        
        



if __name__ == '__main__':
    tracker = Tracker()
    tracker.imu_listener()


