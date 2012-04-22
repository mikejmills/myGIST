import cv2
import numpy
import libgist
import types

class Gist_Processor(object):
    def __init__(self, im, blocks):

    	if isinstance(blocks, numpy.ndarray):
    		self.obj = libgist.GIST_PCA_new(im, blocks)
    	else:	
        	self.obj = libgist.GIST_basic_new(im, blocks)

    def Process(self, im):
    	libgist.GIST_Process(im, self.obj)

    def Get_Descriptor(self, blocks, x=0, y=0):
    	return libgist.GIST_Get_Descriptor_Alloc(blocks, x, y, self.obj)

    def Get_Descriptor_Reuse(self, desc, blocks, x=0, y=0):
    	libgist.GIST_Get_Descriptor_Reuse(desc, blocks, x, y, self.obj)


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