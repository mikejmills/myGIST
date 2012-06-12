#ifndef GIST
#define GIST

#include <vector.h>
#include <pair.h>
#include <map.h>
#include <cv.h>
#include <fftw3.h>
#include <stdio.h>
#include "PCA.h"
#include "/usr/local/include/opencv2/core/types_c.h"
#include <Python.h>

#define IMAGE_WIDTH  320
#define IMAGE_HEIGHT 100

vector<cv::Mat *> *create_gabor(int nscales,  int *orientations, int width, int height);
double              gist_compare(double *d1, double *d2, int size);
double              gist_compare_angle(double *d1, double *d2, int size);
void               gist_free(double *g);

void format_image(cv::Mat &input, cv::Mat &output);


class Gist_Processor
{
  private:
    
    bool               PCA_ENABLED;
    int                nblocks;
    int                *nx, *ny;
    vector<cv::Mat>    GaborResponses;
    vector<cv::Mat>    GaborResponsesInts;
    vector<cv::Mat *>  *gabors;

  	map<int, pair<cv::PCA *, cv::Mat *> >           pca_map;
    map<int, pair<cv::PCA *, cv::Mat *> >::iterator it;

  	double 		  	    *fx, *fy, *gfc;
  	fftwf_complex 	  *in1, *in2, *out, *gin1, *gin2, *gout1, *gout2;
  	fftwf_plan    	  fft1, ifft1, fft2, ifft2, gfft, gifft;
    
    void    prefilt_init(int width, int height);
    void    gfft_init(int width, int height);
    void    down_N(double *res, cv::Mat &src, int N, int cshift=0, int rshift=0);
    void    down_N_rectangle(double *res, cv::Mat &src, int N, int width, int xshift=0, int xshift=0);
  	cv::Mat prefilt_process(cv::Mat &im, int fc);
    void    init(cv::Mat &baseim, int max_blocks);


  public:
    bool PCA_Test() { return PCA_ENABLED; }
    int base_descsize;
  	Gist_Processor(cv::Mat &baseim, int blocks);
    Gist_Processor(cv::Mat &baseim, long int *blocks, int len);
  	~Gist_Processor();

    int   Get_Descriptor(double **res, int block, int xshift=0, int yshift=0);
    int   Get_Descriptor_Rectangle(double **res, int blocks, int width, int xshift=0, int yshift=0);
    void  Get_Descriptor_Rectangle(double *res, int blocks, int width, int xshift=0, int yshift=0);

    void  Get_Descriptor(double *res, int blocks, int xshift=0, int yshift=0);
    void  Get_Descriptor_PCA(double *res, int blocks, int xshift=0, int yshift=0);
    int   Get_Descriptor_PCA(double **res, int blocks, int xshift, int yshift);
    int   Get_Size(int blocks);
    int   Get_Gabors() { return gabors->size(); };

    void Process(cv::Mat &im);
    
    void Save_Descriptor(FILE *fd, double *desc, int size) 
    {
        fprintf(fd, "%d\n", size);
        
        for (int i=0; i < size; i++) {
            fprintf(fd, "%lf\n", desc[i]);
        }

    }

    int Load_Descriptor(FILE *fd, double **desc)
    {
       int    size;
       

       fscanf(fd, "%d", &size);

       *desc = (double *)malloc(sizeof(double) * size);

       for (int i=0; i < size; i++) {
           fscanf(fd, "%lf", &( (*desc)[i] ) );

       }

       return size;
    }
    
};

#endif