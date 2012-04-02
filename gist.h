#ifndef GIST
#define GIST

#include <vector.h>
#include <pair.h>
#include <map.h>
#include <cv.h>
#include <fftw3.h>
#include <stdio.h>
#include "PCA.h"

vector<cv::Mat *> *create_gabor(int nscales,  int *orientations, int width, int height);
float              gist_compare(float *d1, float *d2, int size);
float              gist_compare_angle(float *d1, float *d2, int size);
void               gist_free(float *g);

void format_image(cv::Mat &input, cv::Mat &output);

class Gist_Processor
{
  private:
    
    int                nblocks;
    int                *nx, *ny;
    vector<cv::Mat>    GaborResponses;
    vector<cv::Mat *>  *gabors;
  	map<int, pair<cv::PCA *, cv::Mat *> > pca_map;
    map<int, pair<cv::PCA *, cv::Mat *> >::iterator it;

  	float 		  	    *fx, *fy, *gfc;
  	fftwf_complex 	  *in1, *in2, *out, *gin1, *gin2, *gout1, *gout2;
  	fftwf_plan    	  fft1, ifft1, fft2, ifft2, gfft, gifft;
    
    void    prefilt_init(int width, int height);
    void    gfft_init(int width, int height);
    void    down_N(float *res, cv::Mat &src, int N, int cshift=0, int rshift=0);
  	cv::Mat prefilt_process(cv::Mat &im, int fc);
    void    init(cv::Mat &baseim, int max_blocks);


  public:
    int base_descsize;
  	Gist_Processor(cv::Mat &baseim, int blocks);
    Gist_Processor(cv::Mat &baseim, int *blocks, int len);
  	~Gist_Processor();

    int   Get_Descriptor(float **res, int block, int xshift=0, int yshift=0);
    void  Get_Descriptor(float *res, int blocks, int xshift=0, int yshift=0);
    void  Get_Descriptor_PCA(float *res, int blocks, int xshift=0, int yshift=0);
    int   Get_Size(int blocks);

    void Process(cv::Mat &im);
    
    void Save_Descriptor(FILE *fd, float *desc, int size) 
    {
        fprintf(fd, "%d\n", size);
        
        for (int i=0; i < size; i++) {
            fprintf(fd, "%f\n", desc[i]);
        }

    }

    int Load_Descriptor(FILE *fd, float **desc)
    {
       int    size;
       

       fscanf(fd, "%d", &size);

       *desc = (float *)malloc(sizeof(float) * size);

       for (int i=0; i < size; i++) {
           fscanf(fd, "%f", &( (*desc)[i] ) );

       }

       return size;
    }
    
};


#endif