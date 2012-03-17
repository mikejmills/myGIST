

#include "gist.h"
#include <opencv2/highgui/highgui.hpp>
#include <ctime>

void format_image(cv::Mat &input, cv::Mat &output)
{
	cv::Mat gray, tmp;

	cv::resize(input, output, cv::Size(320,240));
	cv::cvtColor(output, input, CV_BGR2GRAY);
	input.convertTo(output, CV_32FC1, (float)1/255);
}


int main() 
{
	
	cv::Mat input, output;
	cv::namedWindow("test", 1);
	cv::VideoCapture cap("/Users/mike/code/DenseSamplingHoming/video.mp4");

	cap >> input;

	
	float *res;
	int res_size=0;
	format_image(input, output);

	Gist_Processor proc(input, 4);	
	
	clock_t start, end;
	while(1) {
		
		cap >> input;
		
		format_image(input, output);

		start = clock();
		proc.Process(output);
		end = clock();

		printf("Proc time %f\n", float(end - start) / CLOCKS_PER_SEC);

		res_size = proc.Get_Descriptor(&res, 4);
		printf("DESCSIZE %d\n", res_size);
		/*for (int i=0; i < res_size; i++) {
			printf("%f ", res[i]);
		}

		printf("\n");
	*/
		cv::imshow("test", output);
		cv::waitKey(1);
		
	}
	
	return 0;
}
