#!/usr/bin/python

import libgist as gist
import cv2
import numpy as np


cv2.namedWindow("Training Images")
#im = cv2.imread("/Users/mike/Downloads/")
im = cv2.imread("./mec2.jpg")

print im.shape
gist.init(im.shape[1],im.shape[0])
#(pc, desc_dim) = gist.init_pca(mean, eigens)
#i = im.flatten()
#for x in xrange(0,200):
#    print i[x*100:x*100+100]

gist.process(im)

desc = gist.alloc(4,4)
#pca_desc = np.empty([pc,1])

gist.get(desc, -im.shape[1]/2, -im.shape[0]/2, im.shape[1], im.shape[0])

#print "Descr Shape", desc[0].shape, "PCA Shape", pca_desc.shape, "Mean Shape", mean.shape, "Eigen Shape", eigens.shape

#gist.pca_project(desc[0], pca_desc)
print desc[0][0][0:10]
#print desc	