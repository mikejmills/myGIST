#ifndef GIST
#define GIST

#include <vector.h>
#include <cv.h>
#include <fftw3.h>

vector<cv::Mat *> *create_gabor(int nscales,  int *orientations, int width, int height);
float gist_compare(float *d1, float *d2, int size);
float gist_compare_angle(float *d1, float *d2, int size);

class Gist_Processor
{
  private:
    
    vector<cv::Mat>   GaborResponses;
    int               nblocks;
    int               *nx, *ny;
  	vector<cv::Mat *> *gabors;
  	
  	float 		  	    *fx, *fy, *gfc, *buff;
  	fftwf_complex 	  *in1, *in2, *out, *gin1, *gin2, *gout1, *gout2;
  	fftwf_plan    	  fft1, ifft1, fft2, ifft2, gfft, gifft;
    
    void    prefilt_init(int width, int height);
    void    gfft_init(int width, int height);
    void    down_N(float *res, cv::Mat &src, int N, int cshift=0, int rshift=0);
  	cv::Mat prefilt_process(cv::Mat &im, int fc);

  public:

  	Gist_Processor(cv::Mat &baseim, int blocks);
  	~Gist_Processor();

    int Get_Descriptor(float **res, int block, int xshift=0, int yshift=0);
    
    void Process(cv::Mat &im);

};


#endif