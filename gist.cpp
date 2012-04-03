#include "gist.h"

void format_image(cv::Mat &input, cv::Mat &output)
{
    cv::Mat gray, tmp;

    cv::resize(input, output, cv::Size(320,240));
    cv::cvtColor(output, input, CV_BGR2GRAY);
    input.convertTo(output, CV_64FC1, (double)1/255);
}

//
// Swaps the quadrants of the fft so the zero frequency is in the center
static void fftshift(double *data, int w, int h)
{
    int i, j;

    double *buff = (double *) malloc(w*h*sizeof(double));

    memcpy(buff, data, w*h*sizeof(double));

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
        (*Gfs)[i] = new cv::Mat(height, width, CV_64FC1);
    }
    
    double **param = (double **) malloc(nscales * nfilters * sizeof(double *));

    for(i = 0; i < nscales * nfilters; i++) {
        param[i] = (double *) malloc(4*sizeof(double));
    }

    double *fx = (double *) malloc(width*height*sizeof(double));
    double *fy = (double *) malloc(width*height*sizeof(double));
    double *fr = (double *) malloc(width*height*sizeof(double));
    double *f  = (double *) malloc(width*height*sizeof(double));

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
            fx[j*width + i] = (double) i - (double)width/2.0f;
            fy[j*width + i] = (double) j - (double)height/2.0f;

            fr[j*width + i] = sqrtf(fx[j*width + i]*fx[j*width + i] + fy[j*width + i]*fy[j*width + i]);
            f[j*width + i]  = atan2f(fy[j*width + i], fx[j*width + i]);
            
        }
    }

    fftshift(fr, width, height);
    fftshift(f, width, height);
    
    int   cols = ((*Gfs)[0])->cols;

    for(fn = 0; fn < nfilters; fn++)
    {
        double *f_ptr  = f;
        double *fr_ptr = fr;

       // printf("Filter %d %f %f\n", fn, param[fn][3], *f_ptr);

        double *data = (double *)((*Gfs)[fn])->data;
        
        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++)
            {
                double tmp = *f_ptr++ + param[fn][3];

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
void Gist_Processor::init(cv::Mat &baseim, int max_blocks)
{
    int nscales = 3;
    int orientations[3] = {8,8,4};
    nblocks = max_blocks;

    gabors = create_gabor(nscales,  orientations, baseim.cols, baseim.rows);
    
    prefilt_init(baseim.cols, baseim.rows);
    gfft_init(baseim.cols, baseim.rows);

    
    nx   =   (int *) malloc((max_blocks+1)*sizeof(int));
    ny   =   (int *) malloc((max_blocks+1)*sizeof(int));

    for(int i=0;i<nscales;i++) {
        for (int j=0; j < orientations[i]; j++) 
            GaborResponses.push_back(cv::Mat(baseim.rows, baseim.cols, CV_64FC1));
    }

    base_descsize = max_blocks*max_blocks*gabors->size();
    PCA_ENABLED = false;
}


Gist_Processor::Gist_Processor(cv::Mat &baseim, int max_blocks)
{
    init(baseim, max_blocks);
}

Gist_Processor::Gist_Processor(cv::Mat &baseim, int *blocks, int len)
{
    init(baseim, blocks[len-1]);
    int size;
    
    for (int i=0; i < len; i++) {
        size = blocks[i]*blocks[i]*gabors->size();
        pca_map[blocks[i]] = make_pair(PCA_LoadData(blocks[i]), new cv::Mat(1, size, CV_64FC1));
    }

    PCA_ENABLED = true;
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

     for ( it=pca_map.begin() ; it != pca_map.end(); it++ ) {
        PCA_Free((*it).second.first);
        delete (*it).second.second;
    }

}

void Gist_Processor::down_N(double *res, cv::Mat &src, int N, int cshift, int rshift)
{
    int i, j, k, l;
    
    for(i = 0; i < N+1; i++)
    {
        if (cshift > 0) {
            nx[i] = (i*(src.cols-cshift)/(N)) + cshift;
            ny[i] = (i*(src.rows-rshift)/(N)) + rshift;
        } else {
            nx[i] = (i*(src.cols+cshift)/(N));
            ny[i] = (i*(src.rows+rshift)/(N));
        }
        
    }
   
    
    for(l = 0; l < N; l++)
    {
        for(k = 0; k < N; k++)
        {
            double mean = 0.0f;

            for(j = ny[l]; j < ny[l+1]; j++)
            {
                for(i = nx[k]; i < nx[k+1]; i++) {

                    mean += ((double *)src.data)[j*src.cols+i];

                }
            }
            
            double denom = (double)(ny[l+1]-ny[l])*(nx[k+1]-nx[k]);

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
    
    fx  = (double *) fftwf_malloc(width*height*sizeof(double));
    fy  = (double *) fftwf_malloc(width*height*sizeof(double));
    gfc = (double *) fftwf_malloc(width*height*sizeof(double));

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
    double s1 = fc/sqrt(log(2));
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            in1[j*width + i][0] = ((double *)pim.data)[j*width+i];
            in1[j*width + i][1] = 0.0f;

            fx[j*width + i] = (double) i - width/2.0f;
            fy[j*width + i] = (double) j - height/2.0f;

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
            ((double *)pim.data)[j*pim.cols+i] -= in2[j*width+i][0] / (width*height);

            in1[j*width + i][0] = ((double *)pim.data)[j*pim.cols+i] * ((double *)pim.data)[j*pim.cols+i];
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
            ((double *)pim.data)[j*pim.cols+i] = ((double *)pim.data)[j*pim.cols+i] / (0.2f+sqrt(sqrt(in2[j*width+i][0]*in2[j*width+i][0]+in2[j*width+i][1]*in2[j*width+i][1]) / (width*height)));
        }
    }
    
    //
    // Remove borders
    cv::Mat res = pim(cv::Rect(5, 5, pim.cols-10, pim.rows-10));
    
    return res;
}


void Gist_Processor::Process(cv::Mat &im)
{
    int height = im.rows;
    int width  = im.cols;

    prefilt_process(im, 5);

    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            gin1[j*width + i][0] = ((double*)im.data)[j*im.cols+i];
            gin1[j*width + i][1] = 0.0f;
        }
    }

    fftwf_execute(gfft);

    for (int k=0; k < gabors->size(); k++) {


        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++)
            {
                double *data = (double *)((*gabors)[k]->data);

                gout2[j*width+i][0] = gout1[j*width+i][0] * data[j*(*gabors)[k]->cols+i];
                gout2[j*width+i][1] = gout1[j*width+i][1] * data[j*(*gabors)[k]->cols+i];
            }
        }

        fftwf_execute(gifft);

        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++) {
                //((double*)im.data)[j*im.cols+i] 
                ((double*)GaborResponses[k].data)[j*GaborResponses[k].cols+i] = sqrt(gin2[j*width+i][0]*gin2[j*width+i][0]+gin2[j*width+i][1]*gin2[j*width+i][1])/(width*height);
            }
        }

        
    }
    
}



int Gist_Processor::Get_Descriptor(double **res, int blocks, int xshift, int yshift)
{
    if (PCA_ENABLED) 
        return Get_Descriptor_PCA(res, blocks, xshift, yshift);
    

    int size = blocks*blocks*gabors->size();

    *res = (double *) malloc(size*sizeof(double));

    for (int k=0; k < gabors->size(); k++) {
        down_N(*res+k*blocks*blocks, GaborResponses[k], blocks, xshift, yshift);
    }

    return size;
}

int Gist_Processor::Get_Size(int blocks)
{
    return blocks*blocks*gabors->size();
}


void Gist_Processor::Get_Descriptor(double *res, int blocks, int xshift, int yshift)
{
    if (PCA_ENABLED) {
        Get_Descriptor_PCA(res, blocks, xshift, yshift);
        return;
    }

    for (int k=0; k < gabors->size(); k++) {
        down_N(res+k*blocks*blocks, GaborResponses[k], blocks, xshift, yshift);
    }

}


void Gist_Processor::Get_Descriptor_PCA(double *res, int blocks, int xshift, int yshift)
{
    
    cv::Mat Output(1, PCA_DIM, CV_64FC1, (void *)res);

    double *ptr = (double *)pca_map[blocks].second->data;

    for (int k=0; k < gabors->size(); k++) {
        down_N(ptr+k*blocks*blocks, GaborResponses[k], blocks, xshift, yshift);
    }
    
    pca_map[blocks].first->project(*(pca_map[blocks].second), Output);

}

int Gist_Processor::Get_Descriptor_PCA(double **res, int blocks, int xshift, int yshift)
{

    *res = (double *) malloc(PCA_DIM*sizeof(double));
    cv::Mat Output(1, PCA_DIM, CV_64FC1, (void *)(*res));

    double *ptr = (double *)pca_map[blocks].second->data;

    for (int k=0; k < gabors->size(); k++) {
        down_N(ptr+k*blocks*blocks, GaborResponses[k], blocks, xshift, yshift);
    }
    
    pca_map[blocks].first->project(*(pca_map[blocks].second), Output);
    printf("HERE %f\n", Output.at<double>(0,0));
    return PCA_DIM;
}


double gist_compare(double *d1, double *d2, int size)
{
    double sum=0;
    double v;

    for (int i=0; i < size; i++) {
        v = d1[i] - d2[i];
        sum += v*v;
    }
    
    return sqrtf(sum);
    
}

double gist_compare_angle(double *d1, double *d2, int size)
{
    double sum=0, mag1, mag2;
    

    
    for (int i=0; i < size; i++) {
        mag1 +=  d1[i] * d1[i];
        mag2 +=  d2[i] * d2[i];
    }
    
    mag1 = sqrtf(mag1);
    mag2 = sqrtf(mag2);

    for (int i=0; i < size; i++) {
        sum += d1[i] * d2[i];
    }
    
    return acosf(sum/(mag1*mag2)) * 180.0 / 3.14159265;
       
}

void gist_free(double *g)
{
    free(g);
}


