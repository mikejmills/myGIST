

#include "gist.h"
#include <opencv2/highgui/highgui.hpp>
#include <ctime>



int main() 
{
	
	cv::Mat input, output;
	cv::namedWindow("test", 1);
	//cv::VideoCapture cap("/Users/mike/code/DenseSamplingHoming/video.mp4");
	//cap >> input;
	input = cv::imread("test.png");
	
	float res[PCA_DIM], center[PCA_DIM];
	int res_size=0;
	int blocks[3] = {4, 8, 10};

	format_image(input, output);
	Gist_Processor proc(output, blocks, 3);	
	//Get_Descriptor_PCA(float *res, int blocks, int xshift, int yshift)

	proc.Process(output);
	proc.Get_Descriptor_PCA(center, 4, 0, 0);
	
	//for (int i=-160; i < 160; i++) {
	//	res_size = proc.Get_Descriptor_PCA(&res, 4, i);
	//}
	
	/*	
	cv::imshow("test", output);
	cv::waitKey(1);
	*/
		
	
	
	return 0;
}
