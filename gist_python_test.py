#!/usr/bin/python

import libgist as gist
import cv2
import numpy as np

im = cv2.imread("../lear_gist-1.2/image136.jpeg")



print im.shape
gist.init(im.shape[1],im.shape[0])


gist.process(im)
desc = gist.alloc(4,4)

gist.get(desc, 0, im.shape[1])
print desc	
print len(desc[0])