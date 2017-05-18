#!/usr/bin/env python

#AUTHORS: WOLFE MAGNUS, PATRICK KELLEY

import rospy
import numpy as np
import scipy as sp
from scipy.linalg import norm
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from filterpy.monte_carlo import systematic_resample

from particle import ParticleFilter

N_PARTICLES = 10

class Tracker:
    def __init__(self):
        # Init publishers
        self.pub_raw_im       = rospy.Publisher('tracker/raw_im',       Image, queue_size=50)
        self.pub_thresh_sub   = rospy.Publisher('tracker/thresh_sub',   Image, queue_size=50)
        self.pub_ptcls        = rospy.Publisher('tracker/ptcls',        Image, queue_size=50)
    
        # Init BG Subtractors and cv_bridge
        self.fgbg      = cv2.createBackgroundSubtractorMOG2()
        self.fgbg_edge = cv2.createBackgroundSubtractorMOG2()
        self.bridge    = CvBridge()
    
        # Init opflow
        self.prev_cv_im = None
        self.initial_flag = None
        self.old_of = None

        self.p0 = None
        # Params for Shi/Tomasi corner detection
        self.feature_params = dict( maxCorners = 10,
                               qualityLevel = 0.0001,
                               minDistance = 5,
                               blockSize = 11 )
        # Params for Lucas/Kanade optical flow
        self.lk_params = dict( winSize  = (17, 17),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.pfilter = ParticleFilter()
        self.particles = [None]
        self.old_particles = self.particles
        # self.ptcl_dist = 20
        self.weights = np.zeros(N_PARTICLES)
        # self.sensor_std_err = 0.1
        self.xs = []
        
    
    def imu_listener(self):
        # Init ROS node
        rospy.init_node('pf_tracker')

        # Subscribe/Spin
        self.sub = rospy.Subscriber("cam0/image_raw", Image, self.track)
        # input("Press Enter when the buggy enters the frame...")
        rospy.spin()

    def track(self, msg):
        try:
            # Image preprocessing - ROS -> OPENCV bridging and downscaling
            cv_im      = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_im = cv2.blur(cv_im, (3,3))
            # cv_im_down = cv_im
            cv_im_down = self.downscale(cv_im)
            # cv_im_down = self.downscale(cv_im_down)
            # cv_im_down = cv_im

            #PIPELINE_01: Optical Flow Image
            #of_im is visual representation of opflow
            #of_raw is the raw optical flow vectors
            of_raw = self.opflow(cv_im_down)
            of_im  = self.of_image_conv(of_raw)

            #PIPELINE_02: Edge Image of the Optical Flow
            # of_edge_im     = (cv2.Canny(of_im,50,100))
            # of_edge_im_col = cv2.cvtColor(of_edge_im, cv2.COLOR_GRAY2BGR)
        
            #PIPELINE_03: Mask Optical Flow Image
            # Regular background subtraction
            fg_im, bg_im, fg_mask, bg_mask = self.bg_subtract(cv_im_down, self.fgbg)
            # print fg_im[0]

            # Edge-based background subtraction
            # (fg_edge_im, bg_edge_im, fg_edge_mask, bg_edge_mask) = self.bg_subtract(cv2.Canny(cv_im,100,200), self.fgbg_edge)
            # edge_im = cv2.Canny(cv_im_down,100,200)
            # (fg_edge_im, bg_edge_im, fg_edge_mask, bg_edge_mask) = self.bg_subtract(edge_im, self.fgbg_edge)
             
            # print np.sum(edge_im > 0)
            # of_mask_raw = of_raw[edge_im > 0]# np.logical_and(of_raw, edge_im)
            # of_mask_im  = cv2.bitwise_and(of_im, of_im, mask=edge_im)

            #PIPELINE_04: Optical Flow Mean Calculation
            # mean_opFlow = np.mean(of_raw, axis=0)

            #PIPELINE_05: Optical Flow Mean Subtraction
            # mean_sub = self.mean_sub(of_raw, mean_opFlow)
            
            #PIPELINE_06: Threshold Mean-Subtracted Image
            # thresh_sub_im = self.threshold_opFlow(mean_sub, mean_opFlow, fg_mask)

            #PIPELINE_07: Re-Sample Particles
            ptcl_im = self.resample(cv_im_down, fg_mask)
            
            # Print particles
            for p in self.particles:
                cv2.circle( ptcl_im,
                                (p[0].astype('int8'), p[1].astype('int8')),
                                2,
                                (0,0,255),
                                -1)

        except CvBridgeError as e:
            print e

        try:
            # self.publish(fg_im, thresh_sub_im, ptcl_im)
            self.publish(fg_im, cv_im, ptcl_im)

        except CvBridgeError as e:
            print e
            
    def n_not_equal(self, A, B):
        flag = np.array([np.not_equal(a, b) for a, b in zip(A,B)]).ravel()
        return np.any( flag )
    
    def resample(self, img, thresh_im):
        # If particles are unset, initialize particle filter
        if self.particles[0] == None:
            print "Sampling Initial Particles"
            self.particles = self.pfilter.create_uniform_particles((0,img.shape[0]), (0,img.shape[1]), (0, 2*np.pi), N_PARTICLES)
            self.old_particles = self.particles
            (good_new, good_old) = self.getFeaturePoints(img, thresh_im) #TODO: Check feature point coords
        else:
            (good_new, good_old) = self.getFeaturePoints(img, thresh_im) #TODO: Check feature point coords

        # Get feature points
        if good_new != None and len(good_new) > 1:
            good_new = good_new.reshape(-1,2)
            
            # Predict
            mean    = np.mean(good_new, axis=0) #TODO: Check these params too
            std_dev = np.std( good_new, axis=0) #TODO: Probably need to exclude outside of fg mask
            self.pfilter.predict(self.particles, u=mean, std=std_dev)

            # Update
            std_err = np.std(good_new, axis=0)
            zs = np.zeros((len(good_new), len(self.particles)))
            for f_ind, feature in enumerate(good_new):
                for p_ind, particle in enumerate(self.particles[:,0:2]):
                    zs[f_ind,p_ind] += np.linalg.norm(particle - feature)
                

            self.pfilter.update(self.particles,
                                    self.weights,
                                    z=zs, #TODO: Cross check this with example. Especially dims
                                    R=np.array(std_err), #TODO: Play with this(maybe remove it?)
                                    landmarks=good_new)


            # Resample if too few effective particles
            if self.pfilter.neff(self.weights) < N_PARTICLES/2:
                indices = systematic_resample(self.weights)
                self.pfilter.resample(self.particles, self.weights, indices)


            # Estimate position
            mu, var = self.pfilter.estimate(self.particles, self.weights)
            self.xs.append(mu)

        
        ptcl_im = img.copy()
        if good_new != None and len(good_new) > 1:
            for f in good_new:
                cv2.circle( ptcl_im,
                                (f[0].astype('int8'), f[1].astype('int8')),
                                2,
                                (0,255,0),
                                -1)


        return ptcl_im


    def getFeaturePoints(self, frame, mask_im):

        # If initial points are unset, initialize optical flow
        if self.p0 == None:
            print "Sampling Initial Features"
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.old_frame = frame
            old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
            self.p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
        else:
            # Set frames to use for feature points
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)

        # Calculate LK optical flow from Shi/Tomasi features
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0, None, **(self.lk_params))

        # If there aren't good features, make new ones
        if np.all(st==0) or p1 == None:
            # print "No good old features(2)"
            tmp_features = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
            if np.all(tmp_features != None):
                # print "There are good new features!!(2)"
                self.p0 = tmp_features.reshape(-1,2)
            return (self.p0, self.p0)

        # Select good points
        p1 = p1.reshape(-1, 1, 2)
        good_new = np.array(p1[st==1])
        good_old = np.array(self.p0[st==1])
        
        # Now update the previous frame and previous points
        self.old_frame = frame.copy()
        self.p0 = good_new.reshape(-1,1,2)


        # If there aren't good features, make new ones
        if good_new == None or len(good_new) == 0:
            tmp_features = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
            if np.all(tmp_features != None):
                self.p0 = tmp_features.reshape(-1,1,2)
            return (self.p0, self.p0)


        good_new = np.array(good_new)
        inbounds_arr = np.array([np.less(p, np.array(mask_im.shape)) for p in good_new])
        bound_mask =  np.logical_and(inbounds_arr[:,0], inbounds_arr[:,1])
        good_new = good_new[bound_mask]
        good_old = good_old[bound_mask]
        passed_mask = np.where([mask_im[int(p[0]), int(p[1])] > 0 for p in good_new])
        good_new = good_new[passed_mask]
        good_old = good_old[passed_mask]

        # If there aren't good features, make new ones
        if len(good_old) == 0:
            tmp_features = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
            if np.all(tmp_features != None):
                self.p0 = tmp_features.reshape(-1,1,2)
            return (self.p0, self.p0)
        return (good_new.reshape(-1,1,2), good_old)

    def mean_sub(self, vec_img, mean):
        of_sub_raw = np.subtract(vec_img, mean)
        of_sub_mag = np.linalg.norm(of_sub_raw, axis=2)
        of_sub_im = np.multiply( of_sub_mag.astype('uint8'), 10)
        of_sub_im = np.minimum(of_sub_im, 255)

        of_mag = np.linalg.norm(vec_img, axis=2)
        of_im2 = np.multiply( of_mag.astype('uint8'), 10)
        of_im2 = np.minimum(of_im2, 255)

        return of_sub_im
        
    def threshold_opFlow(self, img, mean, fg_mask):
        # thresh_sub_mat = cv2.threshold(img, 20, 255, cv2.THRESH_TOZERO)
        thresh_sub_mat = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        # thresh_sub_im = self.bridge.imgmsg_to_cv2(thresh_sub_mat, "bgr8")
        thresh_otsu = thresh_sub_mat[1]
        thresh = cv2.bitwise_and(thresh_otsu, fg_mask)
        thresh_sub_im = cv2.bitwise_and(img, img, mask=thresh)
        return thresh_sub_im


    def bg_subtract(self, im, bg_subtractor):
        fg_mask = bg_subtractor.apply(im)
        bg_mask = cv2.bitwise_not(fg_mask)
        fg_im   = cv2.bitwise_and(im,im,mask = fg_mask)
        bg_im   = cv2.bitwise_and(im,im,mask = bg_mask)
        return (fg_im, bg_im, fg_mask, bg_mask)

    #downscaling of an image by 50% in both dimensions (gaussian)
    def downscale(self, opencv_image):
        return cv2.pyrDown( cv2.pyrDown(opencv_image) )

    #optical flow computation
    def opflow(self, next_image):
        next_cv_im = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
        if self.prev_cv_im == None:
            self.prev_cv_im = next_cv_im
        if self.old_of == None:
            initial_flag = 0
        else:
            initial_flag = cv2.OPTFLOW_USE_INITIAL_FLOW
            #of_val = cv2.optflow.calcOpticalFlowSF(next_image, next_image, 3, 2, 10)
            #of_val = cv2.optflow.calcOpticalFlowSF(self.prev_cv_im, next_image, 3, 2, 10)
        of_val = cv2.calcOpticalFlowFarneback(self.prev_cv_im,next_cv_im, self.old_of,
                                                  .5,  # Pyr Scale
                                                  3,   # Pyr levels
                                                  15,   # Window size
                                                  2,   # Iterations
                                                  7,   # Pixel neighborhood
                                                  1.1, # Polysigma
                                                  initial_flag)   # Flags
        self.prev_cv_im = next_cv_im
        self.old_of = of_val
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

    def publish(self, fg_im, thresh_sub_im, ptcl_im): # )
        # Publish streams
        # thresh_sub_vis = cv2.cvtColor(thresh_sub_im, cv2.COLOR_GRAY2BGR)
            
        self.pub_raw_im.publish(      self.bridge.cv2_to_imgmsg(fg_im,           "bgr8"))
        self.pub_thresh_sub.publish(  self.bridge.cv2_to_imgmsg(thresh_sub_im,   "bgr8")) #TODO
        self.pub_ptcls.publish(       self.bridge.cv2_to_imgmsg(ptcl_im,         "bgr8")) #TODO
        

if __name__ == '__main__':
    tracker = Tracker()
    tracker.imu_listener()


