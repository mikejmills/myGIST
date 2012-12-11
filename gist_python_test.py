#!/usr/bin/python

import libgist as gist
import cv2
import numpy as np




cv2.namedWindow("Training Images")

im = cv2.imread("./mec2.jpg")


print im.shape
gist.init(im.shape[1],im.shape[0])


gist.process(im)

desc = gist.alloc(4,4)
print desc
gist.get(desc, 0,0, 240, 180)

gist.create_corner_descriptor(im, 5, 10, 10, 2, 2)



