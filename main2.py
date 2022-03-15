# Implementation using ORB Feature Detector & BFMatcher for feature matching
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Here we load image and for accurate results we load both image as a grayscale.
img1=cv.imread('images/trainimage.jpeg')
cv.namedWindow('Image1',cv.WINDOW_NORMAL)
cv.imshow('Image1',img1)
img2=cv.imread('images/queryimage.jpeg')
cv.namedWindow('Image2',cv.WINDOW_NORMAL)
cv.imshow('Image2',img2)
img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

#Create the ORB detector
orb = cv.ORB_create()

#Store the keypoints and detectors for both the images.
#orb.detectAndCompute() returns the keypoints(uses FAST Alogrithm) which contains:
    #co-ordinate of the keypoint
    #diameter of the keypoint neighbourhood(used for scale invariance)
    #angle orientation of the keypoint
    #the response by which the most strong keypoints have been selected.
    #octave(pyramid-layer) from which the keypoint is extracted(basically the scale of the image used)
    #object class
#It also computes descriptors(uses Rotated BRIEF Algorithm) where it outputs concatenated vectors of descriptors where each descriptor is a 32-element vector(less than SIFT and SURF)
kp1,des1 = orb.detectAndCompute(img1,None)
a_1 = cv.drawKeypoints(img1,kp1,None,color = (255,0,0))
kp2, des2 = orb.detectAndCompute(img2,None)
a_2 = cv.drawKeypoints(img2,kp2,None,color = (255,0,0))
cv.namedWindow('Keypoints 1',cv.WINDOW_NORMAL)
cv.namedWindow('Keypoints 2',cv.WINDOW_NORMAL)
cv.imshow("Keypoints 1", a_1)
cv.imshow("Keypoints 2",a_2)

#First create BFMatcher object which matches features by comparing descriptors from the first image with descriptors from other.
#BF-Matcher is simple.It takes the descriptor of one feature in first set and return the closest feature in other by some calculations.
bf = cv.BFMatcher_create()

#Find matching points using k-NN algorithm.
matches = bf.knnMatch(des1,des2,k=2)

#Apply ratio test and store all good points as mentioned in paper(SIFT-SURF)
good_points=[]
list_for_hom=[]

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_points.append([m])
        list_for_hom.append(m)

#We keep a threshold number for matching, if matched we store the image points location and they are passed to form perspective transformation.
min_match=5

if len(good_points)>min_match:
    im1_pts=np.float32([kp1[m.queryIdx].pt for m in list_for_hom]).reshape(-1, 1, 2)
    im2_pts=np.float32([kp2[m.trainIdx].pt for m in list_for_hom]).reshape(-1, 1, 2)


#Now we stich image so first step we find the Homography Matrix by applying RANSAC algorithm
#and then calculate the warping transformation based on matched features.
width=img1.shape[1]+img2.shape[1]
height=img1.shape[0]+img2.shape[1]


H,mask= cv.findHomography(im1_pts,im2_pts,cv.RANSAC, 5.0)
result=cv.warpPerspective(img1,H,(width,height))
#Till now we have wrapped the img1 in the same plane as img2 and now we will place img2 in the resulting img
result[0:img2.shape[0],0:img2.shape[1]]=img2

cv.namedWindow('Combined Image',cv.WINDOW_NORMAL)
cv.imshow('Combined Image',result)
while(cv.waitKey(1)!=ord('q')):
    pass
cv.destroyAllWindows()
