#include "gist.h"


//
// Swaps the quadrants of the fft so the zero frequency is in the center
static void fftshift(float *data, int w, int h)
{
    int i, j;

    float *buff = (float *) malloc(w*h*sizeof(float));

    memcpy(buff, data, w*h*sizeof(float));

    for(j = 0; j < (h+1)/2; j++)
    {
        for(i = 0; i < (w+1)/2; i++) {
            data[(j+h/2)*w + i+w/2] = buff[j*w + i];
        }

        for(i = 0; i < w/2; i++) {
            data[(j+h/2)*w + i] = buff[j*w + i+(w+1)/2];
        }
    }

    for(j = 0; j < h/2; j++)
    {
        for(i = 0; i < (w+1)/2; i++) {
            data[j*w + i+w/2] = buff[(j+(h+1)/2)*w + i];
        }

        for(i = 0; i < w/2; i++) {
            data[j*w + i] = buff[(j+(h+1)/2)*w + i+(w+1)/2];
        }
    }

    free(buff);
}



vector<cv::Mat *> *create_gabor(int nscales,  int *orientations, int width, int height)

{
    int i, j, fn;
    int nfilters = 0;
    
    for(i=0;i<nscales;i++)  
        nfilters += orientations[i];
    


    vector<cv::Mat *> *Gfs = new vector<cv::Mat *>(nfilters);
    for (int i=0; i < nfilters; i++) {
        (*Gfs)[i] = new cv::Mat(height, width, CV_32FC1);
    }
    
    float **param = (float **) malloc(nscales * nfilters * sizeof(float *));

    for(i = 0; i < nscales * nfilters; i++) {
        param[i] = (float *) malloc(4*sizeof(float));
    }

    float *fx = (float *) malloc(width*height*sizeof(float));
    float *fy = (float *) malloc(width*height*sizeof(float));
    float *fr = (float *) malloc(width*height*sizeof(float));
    float *f  = (float *) malloc(width*height*sizeof(float));

    int l = 0;

    for(i = 1; i <= nscales; i++)
    {
        for(j = 1; j <= orientations[i-1]; j++)
        {
            param[l][0] = 0.35f;
            param[l][1] = 0.3/powf(1.85f, i-1);
            param[l][2] = 16*powf(orientations[i-1], 2)/powf(32, 2);
            param[l][3] = M_PI/(orientations[i-1])*(j-1);
            l++;
        }
    }

    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            fx[j*width + i] = (float) i - (float)width/2.0f;
            fy[j*width + i] = (float) j - (float)height/2.0f;

            fr[j*width + i] = sqrtf(fx[j*width + i]*fx[j*width + i] + fy[j*width + i]*fy[j*width + i]);
            f[j*width + i]  = atan2f(fy[j*width + i], fx[j*width + i]);
            
        }
    }

    fftshift(fr, width, height);
    fftshift(f, width, height);
    
    int   cols = ((*Gfs)[0])->cols;

    for(fn = 0; fn < nfilters; fn++)
    {
        float *f_ptr  = f;
        float *fr_ptr = fr;

       // printf("Filter %d %f %f\n", fn, param[fn][3], *f_ptr);

        float *data = (float *)((*Gfs)[fn])->data;
        
        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++)
            {
                float tmp = *f_ptr++ + param[fn][3];

                if(tmp < -M_PI) {
                    tmp += 2.0f*M_PI;
                }
                else if (tmp > M_PI) {
                    tmp -= 2.0f*M_PI;
                }
                
                data[j*cols+i] = expf(-10.0f*param[fn][0]*(*fr_ptr/height/param[fn][1]-1)*
                                    (*fr_ptr/width/param[fn][1]-1)-2.0f*param[fn][2]*M_PI*tmp*tmp);
                fr_ptr++;
                
            }
            
            
        }
        
        
    }
    
    for(i = 0; i < nscales * nfilters; i++) {
        free(param[i]);
    }
    free(param);

    free(fx);
    free(fy);
    free(fr);
    free(f);

    return Gfs;
}



//=============================================================================================================



Gist_Processor::Gist_Processor(cv::Mat &baseim, int blocks)
{
    int nscales = 3;
    int orientations[3] = {8,8,4};
    nblocks = blocks;

    gabors = create_gabor(nscales,  orientations, baseim.cols, baseim.rows);

    prefilt_init(baseim.cols, baseim.rows);
    gfft_init(baseim.cols, baseim.rows);

    nx = (int *) malloc((nblocks+1)*sizeof(int));
    ny = (int *) malloc((nblocks+1)*sizeof(int));

    descsize = 0;
    for(int i=0;i<nscales;i++) 
        descsize+=nblocks*nblocks*orientations[i];


}

Gist_Processor::~Gist_Processor()
{
    delete gabors;

    fftwf_free(fx);
    fftwf_free(fy);
    fftwf_free(gfc);
    fftwf_free(in1);
    fftwf_free(in2);
    fftwf_free(out);

    fftwf_free(fft1);
    fftwf_free(ifft1);

    fftwf_free(fft2); 
    fftwf_free(ifft2);

    fftwf_free(gin1);
    fftwf_free(gin2);

    fftwf_free(gout1);
    fftwf_free(gout2);

    fftwf_free(gfft);
    fftwf_free(gifft);

    free(nx);
    free(ny);

}

void Gist_Processor::down_N(float *res, cv::Mat &src, int N)
{
    int i, j, k, l;
    
    for(i = 0; i < N+1; i++)
    {
        nx[i] = i*src.cols/(N);
        ny[i] = i*src.rows/(N);
    }

    for(l = 0; l < N; l++)
    {
        for(k = 0; k < N; k++)
        {
            float mean = 0.0f;

            for(j = ny[l]; j < ny[l+1]; j++)
            {
                for(i = nx[k]; i < nx[k+1]; i++) {
                    mean += ((float *)src.data)[j*src.cols+i];
                }
            }

            float denom = (float)(ny[l+1]-ny[l])*(nx[k+1]-nx[k]);

            res[k*N+l] = mean / denom;
        }
    }
}


void Gist_Processor::gfft_init(int width, int height)
{
    gin1  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    gin2  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    gout1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    gout2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));

    gfft  = fftwf_plan_dft_2d(width, height, gin1, gout1, FFTW_FORWARD, FFTW_ESTIMATE);
    gifft = fftwf_plan_dft_2d(width, height, gout2, gin2, FFTW_BACKWARD, FFTW_ESTIMATE);

}

void Gist_Processor::prefilt_init(int width, int height)
{
    //PF_Whitening, PF_Normalization;
    width  = width  + 10;
    height = height + 10;
    
    fx  = (float *) fftwf_malloc(width*height*sizeof(float));
    fy  = (float *) fftwf_malloc(width*height*sizeof(float));
    gfc = (float *) fftwf_malloc(width*height*sizeof(float));

    in1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    in2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    out = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));

    fft1  = fftwf_plan_dft_2d(width, height, in1, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ifft1 = fftwf_plan_dft_2d(width, height, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);

    fft2  = fftwf_plan_dft_2d(width, height, in1, out, FFTW_FORWARD, FFTW_ESTIMATE);
    ifft2 = fftwf_plan_dft_2d(width, height, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);

}

cv::Mat Gist_Processor::prefilt_process(cv::Mat &im, int fc)
{
    cv::Mat pim;

    //
    // Add padding
    copyMakeBorder(im, pim, 5, 5, 5, 5, IPL_BORDER_CONSTANT, cv::Scalar(0,0,0));
    
    int i,j;
    int width  = pim.cols;
    int height = pim.rows;

    //
    // Build whitening filter and apply whitening filter
    float s1 = fc/sqrt(log(2));
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            in1[j*width + i][0] = ((float *)pim.data)[j*width+i];
            in1[j*width + i][1] = 0.0f;

            fx[j*width + i] = (float) i - width/2.0f;
            fy[j*width + i] = (float) j - height/2.0f;

            gfc[j*width + i] = exp(-(fx[j*width + i]*fx[j*width + i] + fy[j*width + i]*fy[j*width + i]) / (s1*s1));
        }
    }

    fftshift(gfc, width, height);

    fftwf_execute(fft1);

    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            out[j*width+i][0] *= gfc[j*width + i];
            out[j*width+i][1] *= gfc[j*width + i];
        }
    }

    fftwf_execute(ifft1);


    //
    // Local contrast normalisation 
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            ((float *)pim.data)[j*pim.cols+i] -= in2[j*width+i][0] / (width*height);

            in1[j*width + i][0] = ((float *)pim.data)[j*pim.cols+i] * ((float *)pim.data)[j*pim.cols+i];
            in1[j*width + i][1] = 0.0f;
        }
    }

    fftwf_execute(fft2);

    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            out[j*width+i][0] *= gfc[j*width + i];
            out[j*width+i][1] *= gfc[j*width + i];
        }
    }

    fftwf_execute(ifft2);


    //
    // Get result from contrast normalisation filter
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++) {
            ((float *)pim.data)[j*pim.cols+i] = ((float *)pim.data)[j*pim.cols+i] / (0.2f+sqrt(sqrt(in2[j*width+i][0]*in2[j*width+i][0]+in2[j*width+i][1]*in2[j*width+i][1]) / (width*height)));
        }
    }

    //
    // Remove borders
    cv::Mat res = pim(cv::Rect(5, 5, pim.cols-10, pim.rows-10));
    
    return res;
}


int Gist_Processor::process(cv::Mat &im, int fc, float **res)
{
    int height = im.rows;
    int width  = im.cols;

    *res = (float *) malloc(nblocks*nblocks*gabors->size()*sizeof(float));

    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            gin1[j*width + i][0] = ((float*)im.data)[j*im.cols+i];
            gin1[j*width + i][1] = 0.0f;
        }
    }

    fftwf_execute(gfft);

    for (int k=0; k < gabors->size(); k++) {


        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++)
            {
                float *data = (float *)((*gabors)[k]->data);

                gout2[j*width+i][0] = gout1[j*width+i][0] * data[j*(*gabors)[k]->cols+i];
                gout2[j*width+i][1] = gout1[j*width+i][1] * data[j*(*gabors)[k]->cols+i];
            }
        }

        fftwf_execute(gifft);

        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++) {
                ((float*)im.data)[j*im.cols+i] = sqrt(gin2[j*width+i][0]*gin2[j*width+i][0]+gin2[j*width+i][1]*gin2[j*width+i][1])/(width*height);
            }
        }

        down_N(*res+k*nblocks*nblocks, im, nblocks);

    }
    
    return descsize;
}





