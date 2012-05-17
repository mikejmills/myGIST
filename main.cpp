

#include "gist.h"
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include "../GISTHoming/alglib/ap.h"
#include "../GISTHoming/alglib/interpolation.h"


#define CURVE_SAMPLE_SIZE 20
#define SAMPLE_LEN 3

alglib::real_1d_array  sim_points;
alglib::real_1d_array  shift_points;
alglib::real_1d_array  coefficients;

alglib::barycentricinterpolant p;
alglib::polynomialfitreport    rep;


void Get_Samples(Gist_Processor *proc, double *desc_goal, int range_width, int prev_min, int blocks)
{

	int hrange = (range_width/2);
	int shift  = range_width/CURVE_SAMPLE_SIZE;
	double min=FLT_MAX, max=0;

	double *desc_curr;
	
	for (int p=-hrange, si=0; p < hrange, si < CURVE_SAMPLE_SIZE; p+=shift, si++){
		
		int size = proc->Get_Descriptor(&desc_curr, blocks, p + prev_min);
		
		double sim = gist_compare(desc_goal, desc_curr, size);
		
		if (min > sim) min = sim;
		if (max < sim) max = sim;

		sim_points[si]   = sim;
		shift_points[si] = p + prev_min;
		//printf("count %d %d\n", si, range_width/shift);
	}
	
	double range = max - min;

	//printf("range %lf\n", range);
	
	for (int p=-hrange, si=0; p < hrange, si < CURVE_SAMPLE_SIZE; p+=shift, si++){
	//	printf("%lf ", sim_points[si]);
		sim_points[si] = sim_points[si]/range;
		printf("%d %lf\n", p + prev_min, sim_points[si]);
	}
	
}

void Min_Similarity(Gist_Processor *proc, int cols, double **desc_goal, int *min_shift, double *min_sim)
{
	alglib::ae_int_t info, m=3;
	int prev_min = 0;
	int hcols = cols/2;
	double min   = 0;
	
	int range = cols;

	for (int blk_idx=0; blk_idx < SAMPLE_LEN; blk_idx++) {
		
		

		if (prev_min < -hcols) prev_min = -hcols + range/2;
		if (prev_min >  hcols) prev_min = hcols - range/2;

		//printf("Prev Min %d\n", prev_min);
		Get_Samples(proc, desc_goal[blk_idx], range, prev_min, (blk_idx+1) * 4);

		alglib::polynomialfit(shift_points, sim_points, m, info, p, rep);
		alglib::polynomialbar2pow(p, coefficients);
	
		double a = coefficients[2], b = coefficients[1], c = coefficients[0];

		if (a >= 0 and b == 0) {
			min = c;
		}

		if (b != 0 and a > 0) {
			min = c - (b*b)/(4*a);
		} else
			min = 0;

		*min_sim = min;

		//
		// Find Minimum
		if (a > 0) {
			min = -b/(2*a);
		} else
			min = 9999;

		//printf("a %lf b %lf c %lf %lf %lf\n", a, b, c, min_sim, min);

		/*
		printf("shift %d\n", (int) min);
		
		printf("%d %lf %lf %lf %lf\n", (int) min, min, a, b, c);
		*/

		*min_shift = (int)min;
		prev_min   = (int)min;

		range = range/4;
		
	}

	return;
}

int main() 
{
	sim_points.setlength(CURVE_SAMPLE_SIZE);
	shift_points.setlength(CURVE_SAMPLE_SIZE);

	cv::Mat input, output;
	cv::namedWindow("center", 1);
	cv::namedWindow("other", 1);
	//cv::VideoCapture cap("/Users/mike/code/DenseSamplingHoming/video.mp4");
	//cap >> input;
	input = cv::imread("center.jpg");
	
	double *res, *center[3];
	int res_size=0;
	int blocks[3] = {4, 8, 10};
	cv::imshow("center", input);
	format_image(input, output);
	Gist_Processor proc(output, 12);
	//Get_Descriptor_PCA(float *res, int blocks, int xshift, int yshift)

	proc.Process(output);
	res_size = proc.Get_Descriptor(&res, 4,0,0);
	
	/*//proc.Get_Descriptor_PCA(center, 4, 0, 0);
	
	res_size = proc.Get_Descriptor(&center[0], 4, 0);
	res_size = proc.Get_Descriptor(&center[1], 8, 0);
	res_size = proc.Get_Descriptor(&center[2], 12, 0);

	input = cv::imread("../HallwayDataset/image0287.png");
	cv::imshow("other", input);

	format_image(input, output);
	proc.Process(output);
	
	res_size = proc.Get_Descriptor(&res, (1)*4, 0);
	for (int i=0; i < res_size; i++) {	
		printf("%d %lf\n", i, res[i]);
	}

	int min_shift;
	double min_sim;

	/*Min_Similarity(&proc, output.cols, center, &min_shift, &min_sim);
	printf("min_shift %d min_sim %lf\n", min_shift, min_sim);
	for (int b=0; b < 3; b++){
		min_sim = FLT_MAX;
		for (int i=-160; i < 160; i++) {
			res_size = proc.Get_Descriptor(&res, (b+1)*4, i);
			double sim = gist_compare(center[b], res, res_size);
			printf("%d %lf\n", i, sim);
			if (min_sim > sim) { 
				min_sim   = sim;	
				min_shift = i;
			}
		}
		//printf("%d Shift %d min %lf\n", (b+1)*4, min_shift, min_sim);
	}
	
	cv::waitKey(1);
	*/
	
	sleep(60);
	
	return 0;
}
