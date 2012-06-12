#!/usr/bin/python

import libgist as gist
import cv2

im = cv2.imread("center.jpg")

gist.init(320,100)
gist.process(im)
desc = gist.alloc(4,1)

gist.get(desc, 0, 100)
print desc	