#!/usr/bin/python

import libgist as gist
import cv2
import numpy as np


<<<<<<< HEAD
<<<<<<< HEAD
mean = np.load("PCAmean_8x4.npy")
eigens = np.load("PCAeigenvectors_8x4.npy")

mean = np.transpose(mean)

#eigens = np.transpose(eigens)

im = cv2.imread("../lear_gist-1.2/black.jpeg")
#im = cv2.imread("./left.jpg")
=======
cv2.namedWindow("Training Images")
#im = cv2.imread("/Users/mike/Downloads/")
im = cv2.imread("./mec2.jpg")
>>>>>>> c87829bc4cd2d033d45ccf2fad63f87737368747
=======
cv2.namedWindow("Training Images")
#im = cv2.imread("/Users/mike/Downloads/")
im = cv2.imread("./mec2.jpg")
>>>>>>> c87829bc4cd2d033d45ccf2fad63f87737368747

print im.shape
gist.init(im.shape[1],im.shape[0])
#(pc, desc_dim) = gist.init_pca(mean, eigens)
<<<<<<< HEAD
<<<<<<< HEAD
=======
#i = im.flatten()
#for x in xrange(0,200):
#    print i[x*100:x*100+100]
>>>>>>> c87829bc4cd2d033d45ccf2fad63f87737368747
=======
#i = im.flatten()
#for x in xrange(0,200):
#    print i[x*100:x*100+100]
>>>>>>> c87829bc4cd2d033d45ccf2fad63f87737368747

gist.process(im)

desc = gist.alloc(4,4)
#pca_desc = np.empty([pc,1])
<<<<<<< HEAD
<<<<<<< HEAD

gist.get(desc, 0, 10)
print desc
#print "Descr Shape", desc[0].shape, "PCA Shape", pca_desc.shape, "Mean Shape", mean.shape, "Eigen Shape", eigens.shape

#gist.pca_project(desc[0], pca_desc)

#print pca_desc	
=======

gist.get(desc, -im.shape[1]/2, -im.shape[0]/2, im.shape[1], im.shape[0])

#print "Descr Shape", desc[0].shape, "PCA Shape", pca_desc.shape, "Mean Shape", mean.shape, "Eigen Shape", eigens.shape

#gist.pca_project(desc[0], pca_desc)
print desc[0][0][0:10]
#print desc	
>>>>>>> c87829bc4cd2d033d45ccf2fad63f87737368747
=======

gist.get(desc, -im.shape[1]/2, -im.shape[0]/2, im.shape[1], im.shape[0])

#print "Descr Shape", desc[0].shape, "PCA Shape", pca_desc.shape, "Mean Shape", mean.shape, "Eigen Shape", eigens.shape

#gist.pca_project(desc[0], pca_desc)
print desc[0][0][0:10]
#print desc	
>>>>>>> c87829bc4cd2d033d45ccf2fad63f87737368747
