import cv2
import numpy
import libgist
import types

def Get_ColRange(img_width, blks, cshift):


	bwth = img_width/blks
	col = cshift/(bwth)
	h = img_width/2
	print "col", col

	if cshift < h:
		partial = float(cshift - col*bwth)/bwth
		return (range(0,col+1,1), partial)

	if cshift > h:
		partial = float((col+1)*bwth - cshift)/bwth
		return (range(col, blks, 1), partial)

	return (range(0, blks,1), 1)

def Get_ShiftDesc(img_width, blks, desc, c_pixel):
	
	# width between images in descriptor
	wsimg = blks*blks

	(r, partial) = Get_ColRange(img_width, blks, c_pixel)
	ndesc = []

	for i in range(0,20,1):
		for c in r:
		
			tmp = desc[(c*blks)+i*wsimg:((c+1)*blks)+i*wsimg]
		
			if c_pixel < (img_width/2):
				if c == r[len(r)-1]:
					tmp = partial*tmp
			else:
				if c == r[0]:
					tmp = partial*tmp

			ndesc.extend(tmp)
		
	return ndesc

#######################################################################################
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