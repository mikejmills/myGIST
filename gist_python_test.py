#!/usr/bin/python

import libgist as gist
import cv2
import numpy as np


mean = np.load("PCAmean_8x4.npy")
eigens = np.load("PCAeigenvectors_8x4.npy")

mean = np.transpose(mean)

#eigens = np.transpose(eigens)

im = cv2.imread("../lear_gist-1.2/image136black.jpeg")
#im = cv2.imread("./left.jpg")

print im.shape
gist.init(im.shape[1],im.shape[0])
#(pc, desc_dim) = gist.init_pca(mean, eigens)

gist.process(im)

desc = gist.alloc(4,4)
#pca_desc = np.empty([pc,1])

gist.get(desc, -im.shape[1]/2, -im.shape[0]/2, 40, 40)

#print "Descr Shape", desc[0].shape, "PCA Shape", pca_desc.shape, "Mean Shape", mean.shape, "Eigen Shape", eigens.shape

#gist.pca_project(desc[0], pca_desc)

print desc	