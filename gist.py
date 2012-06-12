import cv2
import numpy
import libgist
import types
from numpy import *


#######################################################################################
class Gist_Processor(object):
	def __init__(self):		
		print "Tmp"
		(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.gabors) = libgist.GIST_Get_Info()

	def Process(self, im):
		libgist.GIST_Process(im)

	def Get_Descriptor(self, blocks, x=0, y=0):
		return libgist.GIST_Get_Descriptor_Alloc(blocks, x, y)

	def Get_Descriptor_Rectangle(self, blocks, width, x=0, y=0):
		return libgist.GIST_Get_Descriptor_Rectangle_Alloc(blocks, width, x, y)

	def Get_Descriptor_Rectangle_Reuse(self, desc, blocks, width, x=0, y=0):
		return libgist.GIST_Get_Descriptor_Rectangle_Reuse(desc, blocks, width, x, y)

	def Get_Descriptor_Reuse(self, desc, blocks, x=0, y=0):
		libgist.GIST_Get_Descriptor_Reuse(desc, blocks, x, y)

	def Get_Descriptor_Size(self, blocks):
		return blocks*blocks*self.gabors

    #def Get_Descriptor_Shift(self, desc=None):

    
'''
if __name__=="__main__":
	print "GIST Tests"
	cam = cv2.VideoCapture(0)
	cv2.namedWindow("test")

	
	success, im = cam.read()
	#im = cv2.imread("../cHoming/1.jpg")
	g = Gist_Processor(im, numpy.array([[4]]))
	#g.Process(im)
	#desc  = g.Get_Descriptor(8)
	#print desc
	
	g.Process(im)
	desc  = g.Get_Descriptor(4)

	while 1:
		success, im = cam.read()
		g.Process(im)
		g.Get_Descriptor_Reuse(desc, 4)	
		print desc
		cv2.imshow("test", im)
		cv2.waitKey(1)

	
	
	g.Process(im)

	desc  = g.Get_Descriptor(4)
	print desc


	g.Process(im2)
	g.Get_Descriptor_Reuse(desc, 4) 
	print desc
'''