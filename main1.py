# Implemented using SIFT for  feature detection & BFMatcher for feature matching.
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
MIN_CHECK=5

#Reading images for stiching & basic operations
img1=cv.imread(r"images/trainimage.jpeg",cv.IMREAD_COLOR)
cv.namedWindow('Image1',cv.WINDOW_NORMAL)
cv.imshow('Image1',img1)
img2=cv.imread(r"images/queryimage.jpeg",cv.IMREAD_COLOR)
cv.namedWindow('Image2',cv.WINDOW_NORMAL)
cv.imshow('Image2',img2)
img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

# Creating SIFT detector object & detecting keypoints
s = cv.SIFT_create()
f1,img12 = s.detectAndCompute(img1,None)
a_1 = cv.drawKeypoints(img1,f1,None,color = (255,0,0))
f2,img22 = s.detectAndCompute(img2,None)
a_2 = cv.drawKeypoints(img2,f2,None,color = (255,0,0))
cv.namedWindow('Keypoints 1',cv.WINDOW_NORMAL)
cv.namedWindow('Keypoints 2',cv.WINDOW_NORMAL)
cv.imshow("Keypoints 1", a_1)
cv.imshow("Keypoints 2",a_2)

# Creating BFMatcher object for feature matching
bf = cv.BFMatcher(cv.NORM_L2)
m=bf.knnMatch(img12,img22,k=2)
final=[]
list_for_hom=[]
for a,b in m:
    if ((a.distance/b.distance)<0.75):
        final.append([a])
        list_for_hom.append(a)


if(len(final)>MIN_CHECK):
    src_pts=np.float32([ f1[m.queryIdx].pt for m in list_for_hom]).reshape(-1,1,2)
    des_pts=np.float32([ f2[m.trainIdx].pt for m in list_for_hom]).reshape(-1,1,2)
hom_mat,mask=cv.findHomography(src_pts,des_pts,cv.RANSAC,5)
print(hom_mat)

h=img1.shape[0]+img2.shape[0]
w=img1.shape[1]+img2.shape[1]
img3=cv.warpPerspective(img1,hom_mat,(w,h))
img3[0:img2.shape[0],0:img2.shape[1]]=img2

cv.namedWindow('Combined Image',cv.WINDOW_NORMAL)
cv.imshow('Combined Image',img3)
while(cv.waitKey(1)!=ord('q')):
	pass
cv.imwrite(r"photos/panorama_stitching_output.png",img3)
cv.destroyAllWindows()
