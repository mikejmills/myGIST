#!/usr/bin/python

import libgist as gist
import sys
import os
import cv2, cv
import numpy as np

#./pca_gen.py ./image_dir pc_size xblks yblks

gist.init(320,50)
print "PCA size", int(sys.argv[2])
pcadata = None

xblks=int(sys.argv[3])
yblks=int(sys.argv[4])

print sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

for f in os.listdir(sys.argv[1]):
	desc = gist.alloc(xblks,yblks)
	im = cv2.imread(sys.argv[1] + "/" + f)

	gist.process(im)
	gist.get(desc,0,320)

	if pcadata == None:
		pcadata = desc[0]
	else:
		pcadata = np.hstack([pcadata, desc[0]])
		
pcadata = np.transpose(pcadata)
print pcadata.shape
mean, eigenvectors = cv2.PCACompute(pcadata, maxComponents=int(sys.argv[2]))

print "Primary Component count ", eigenvectors.shape

np.save("PCAmean_" + str(xblks) + "x" + str(yblks), mean)
np.save("PCAeigenvectors_" +  str(xblks) + "x" + str(yblks), eigenvectors)



