

#include "gist.h"
#include <opencv2/highgui/highgui.hpp>


int main() 
{
	
	cv::Mat baseim;
	cv::namedWindow("test", 1);
	cv::VideoCapture cap("/Users/mike/code/DenseSamplingHoming/video.mp4");

	cap >> baseim;

	cv::Mat gray(baseim.rows, baseim.cols, CV_8UC1);
	cv::Mat grayf(baseim.rows, baseim.cols, CV_32FC1);;

	float *res;
	int res_size=0;

	Gist_Processor proc(baseim, 4);	
	

	while(1) {
		
		cap >> baseim;
		
		cv::cvtColor(baseim, gray, CV_BGR2GRAY);
		gray.convertTo(grayf, CV_32FC1, (float)1/255);

		res_size = proc.process(grayf, 5, &res);
		
		for (int i=0; i < res_size; i++) {
			printf("%f ", res[i]);
		}

		printf("\n");

		cv::imshow("test", grayf);
		cv::waitKey(1);
	
	}
	
	return 0;
}