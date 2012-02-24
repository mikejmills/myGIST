#ifndef GIST
#define GIST

#include <vector.h>
#include <cv.h>
#include <fftw3.h>

vector<cv::Mat *> *create_gabor(int nscales,  int *orientations, int width, int height);

class Gist_Processor
{
  private:
    int               nblocks;
    int               *nx, *ny;
  	vector<cv::Mat *> *gabors;
  	
  	float 		  	    *fx, *fy, *gfc;
  	fftwf_complex 	  *in1, *in2, *out, *gin1, *gin2, *gout1, *gout2;
  	fftwf_plan    	  fft1, ifft1, fft2, ifft2, gfft, gifft;
    
    void    prefilt_init(int width, int height);
    void    gfft_init(int width, int height);
    void    down_N(float *res, cv::Mat &src, int N);
  	cv::Mat prefilt_process(cv::Mat &im, int fc);
  	
  public:
    int descsize;

  	Gist_Processor(cv::Mat &baseim, int blocks);
  	~Gist_Processor();

    
    int process(cv::Mat &im, int fc, float **res);

};


#endif