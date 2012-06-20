
#include "python_gist.h"
#include <vector.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <fftw3.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#include <numpy/ndarrayobject.h>


int IMAGE_HEIGHT = 240, IMAGE_WIDTH = 320;
vector<cv::Mat *> *Gabor_filters;
vector<cv::Mat>   *Response_Image;

/*
inline cv::Mat Get_cvMat_From_Numpy_Mat(cv::Mat matin) 
		const npy_intp* _strides = PyArray_STRIDES(matin);
        const npy_intp* _sizes = PyArray_DIMS(matin);
        int    size[CV_MAX_DIM+1];
        size_t step[CV_MAX_DIM+1];
        size[0] = (int)_sizes[0];
        size[1] = (int)_sizes[1];
		step[0] = (size_t) _strides[0];
        step[1] = (size_t) _strides[1];
        cv::Mat tmp(2, size, 16, PyArray_DATA(matin), step); // 16 is the image type using builting functions returns wrong type
        return tmp;
}*/

inline void Get_cvMat_From_Numpy_Mat(PyArrayObject *matin, cv::Mat &output, int type)
{
    const npy_intp* _strides = PyArray_STRIDES(matin);
    const npy_intp* _sizes   = PyArray_DIMS(matin);
    int    size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    
    size[0] = (int)_sizes[0];
    size[1] = (int)_sizes[1];
    step[0] = (size_t) _strides[0];
    step[1] = (size_t) _strides[1];
    //printf("%d %d\n", PyArray_TYPE(matin), NPY_UINT8);
   // exit(1);
    output = cv::Mat(2, size, type, PyArray_DATA(matin), step); // 16 is the image type using builting functions returns wrong type
}
//=============================================================================================================================

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

//=============================================================================================================================
fftwf_complex *gin1, *gin2, *gout1, *gout2;
fftwf_plan     gfft, gifft;

void gfft_init(int width, int height)
{
    gin1  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    gin2  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    gout1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    gout2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));

    gfft  = fftwf_plan_dft_2d(width, height, gin1, gout1, FFTW_FORWARD, FFTW_ESTIMATE);
    gifft = fftwf_plan_dft_2d(width, height, gout2, gin2, FFTW_BACKWARD, FFTW_ESTIMATE);

}



//=============================================================================================================================
double        *fx, *fy, *gfc;
fftwf_complex *in1, *in2, *out;
fftwf_plan     fft1, ifft1, fft2, ifft2;

void prefilt_init(int width, int height)
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

//=============================================================================================================================

vector<cv::Mat> *response_init(int width, int height)
{
	vector<cv::Mat> *response_images =  (vector<cv::Mat> *) new vector<cv::Mat>;

	for (int scl=0; scl < N_SCALES; scl++) {
		for (int ori=0; ori < orientations[scl]; ori++) {
			response_images->push_back(cv::Mat::zeros(height+1, width+1, CV_64FC1)); // +1 for some tricky Integral image stuff later
		}
	}

	return response_images;
}
//=============================================================================================================================
cv::Mat tmp_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_64FC1);

void format_image(cv::Mat &input, cv::Mat &output)
{
    
    cv::resize(input, output, cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT));
    cv::cvtColor(output, tmp_image, CV_BGR2GRAY);
    tmp_image.convertTo(output, CV_64FC1, 1);

}
//=============================================================================================================================
cv::Mat prefilt_process(cv::Mat &im, int fc)
{
    cv::Mat pim;
    int i,j;
    int width  = im.cols;
    int height = im.rows;
    //
    // Log
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++) {
            //((double*)im.data)[j*im.cols+i] = (double)log( ((double*)im.data)[j*im.cols+i]+1.0f );
            im.at<double>(j,i) = log(im.at<double>(j,i)+1.0);
        }
    }
    
    //
    // Add padding
    copyMakeBorder(im, pim, 5, 5, 5, 5, IPL_BORDER_REPLICATE);
    
    width  = pim.cols;
    height = pim.rows;

    //
    // Build whitening filter and apply whitening filter
    double s1 = fc/sqrt(log(2.0));
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            in1[j*width + i][0] = pim.at<double>(j,i);
            in1[j*width + i][1] = 0.0f;
            

            fx[j*width + i] = (double) i - width/2.0;
            fy[j*width + i] = (double) j - height/2.0;

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
            //((double *)pim.data)[j*pim.cols+i] -= in2[j*width+i][0] / (width*height);
            pim.at<double>(j,i) -= in2[j*width+i][0] / (width*height);

            in1[j*width + i][0] = pim.at<double>(j,i) * pim.at<double>(j,i);
            in1[j*width + i][1] = 0.0;

            //printf("local %f\n", in1[j*width + i][0]);
        }
    }
    //exit(1);

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
            pim.at<double>(j,i) = pim.at<double>(j,i) / (0.2+sqrt(sqrt(in2[j*width+i][0]*in2[j*width+i][0]+in2[j*width+i][1]*in2[j*width+i][1]) / (width*height)));
        }
    }
    
    
    //
    // Remove borders
    cv::Mat res = pim(cv::Rect(5, 5, pim.cols-10, pim.rows-10));
    
    return res;
}
//=============================================================================================================================
void Process(cv::Mat &im)
{
    int height = im.rows;
    int width  = im.cols;
    
    im = prefilt_process(im, 4);
    
    
    
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            gin1[j*width + i][0] = im.at<double>(j,i);
    
            gin1[j*width + i][1] = 0.0f;
        }
    }
    

    fftwf_execute(gfft);

    for (int k=0; k < Gabor_filters->size(); k++) {


        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++)
            {
                //double *data = (double *)((*Gabor_filters)[k]->data);
                gout2[j*width+i][0] = gout1[j*width+i][0] * (*Gabor_filters)[k]->at<double>(j,i); //data[j*(*Gabor_filters)[k]->cols+i];
                gout2[j*width+i][1] = gout1[j*width+i][1] * (*Gabor_filters)[k]->at<double>(j,i); //data[j*(*Gabor_filters)[k]->cols+i];
            }
        }

        fftwf_execute(gifft);

        //
        // Process the results into a integral image note the crazy indexes
        int h = height-1;
        int w = width-1;
        for(int j = 0; j < h; j++)
        {
            for(int i = 0; i < w; i++) {
            	
                (*Response_Image)[k].at<double>(j+1, i+1) = (double)sqrt(gin2[j*width+i][0]*gin2[j*width+i][0]+gin2[j*width+i][1]*gin2[j*width+i][1])/(width*height)
                                                            + (*Response_Image)[k].at<double>(j+1,i)
                                                            + (*Response_Image)[k].at<double>(j,i+1)
                                                            - (*Response_Image)[k].at<double>(j,i);
            }
        }
        
        
    }
   
    
}
//=============================================================================================================================
#define MAX_BLOCKS 10
#define DESCRIPTOR_SIZE(x, y) x * y * Gabor_filters->size()

int nx[MAX_BLOCKS], ny[MAX_BLOCKS];


void Fill_Descriptor(double *desc, int xoffset, int win_width, int xblks, int yblks)
{

    int i, x, y;
    
    int cols = IMAGE_WIDTH, rows = IMAGE_HEIGHT;

    int himg   = cols/2;

    int hwin = win_width/2;

    int width  = (win_width/xblks)-1;
    int height = (rows/yblks)-1;

    xoffset = xoffset + himg - hwin;
   
    if (xoffset < 0)                         xoffset = 0;
    if (xoffset > (IMAGE_WIDTH - win_width)) xoffset = IMAGE_WIDTH - win_width;

    nx[0] = xoffset;
    ny[0] = 0;
    
    //printf("xoffset %d\n", xoffset);

    for( i=1; i < xblks+1; i++) {
        nx[i] = (nx[i-1] + width);
    }
    
    for( i=1; i < yblks+1; i++) {
        ny[i] = (ny[i-1] + height);
    }
    

    double denom = (double)(ny[1]-ny[0])*(nx[1]-nx[0]);
   

    for (int gbr = 0; gbr < Response_Image->size(); gbr++) {

        cv::Mat src = (*Response_Image)[gbr];
        double *res = desc+gbr*xblks*yblks;

        for(y = 0; y < yblks; y++) {
            for(x = 0; x < xblks; x++) {  
                double mean   = (  src.at<double>(ny[y+1], nx[x+1]) 
                                 + src.at<double>(ny[y], nx[x]) 
                                 - src.at<double>(ny[y+1], nx[x])
                                 - src.at<double>(ny[y], nx[x+1]) );
                

                res[y*xblks+x]=  mean / denom;
             }
        }
    
    }


}

//=============================================================================================================================
cv::PCA pca_object;
int PCcount = 0;

PyObject *PCA_project(PyObject *obj, PyObject *args)
{
    PyArrayObject  *desc, *pca_desc;
    cv::Mat        cvdesc, cvpca_desc;

    if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &desc, &PyArray_Type, &pca_desc))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }

    Get_cvMat_From_Numpy_Mat(desc, cvdesc, CV_64FC1);
    Get_cvMat_From_Numpy_Mat(pca_desc, cvpca_desc, CV_64FC1);

    /*printf("%d %d : %d %d data %d %d : %d %d\n", cvdesc.cols, cvdesc.rows, cvpca_desc.cols, cvpca_desc.rows,   
                                                 pca_object.mean.cols, pca_object.mean.rows, pca_object.eigenvectors.cols, pca_object.eigenvectors.rows);
    */
    
    pca_object.project(cvdesc, cvpca_desc);



    Py_INCREF(Py_None);
    return Py_None;    

}

//=============================================================================================================================
PyObject *Init_GIST(PyObject* obj, PyObject *args)
{
	int cols, rows;

	if (!PyArg_ParseTuple(args, "ii",  &cols,  &rows))  {
		printf("FAILED PROCESSING Parsing\n");
		return NULL;
	}
	

	Gabor_filters = create_gabor(N_SCALES, orientations, cols, rows);
	prefilt_init(cols, rows);
	gfft_init(cols, rows);
	Response_Image = response_init(cols, rows);

	IMAGE_WIDTH  = cols;
	IMAGE_HEIGHT = rows;

	Py_INCREF(Py_None);
	return Py_None;
}


PyObject *Init_PCA(PyObject* obj, PyObject *args)
{
    PyArrayObject *mean, *eigenvectors;
    

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &mean, &PyArray_Type, &eigenvectors))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }
    
    Get_cvMat_From_Numpy_Mat(mean, pca_object.mean, CV_64FC1);
    Get_cvMat_From_Numpy_Mat(eigenvectors, pca_object.eigenvectors, CV_64FC1);
    
    PCcount = pca_object.eigenvectors.rows;

    
    return Py_BuildValue("(ii)", PCcount, pca_object.mean.cols);

}


PyObject *Process_Image(PyObject *obj, PyObject *args)
{
	cv::Mat        output, tmp;
	PyArrayObject  *imarray;

	if (!PyArg_ParseTuple(args, "O!",  &PyArray_Type,  &imarray))  {
		printf("FAILED PROCESSING Parsing\n");
		return NULL;
	}

	Py_INCREF(imarray);

	Get_cvMat_From_Numpy_Mat(imarray, tmp, 16);
	format_image(tmp, output);
   
    
	Process(output);
	
	Py_DECREF(imarray);
	

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject *Descriptor_Allocate(PyObject *obj, PyObject *args)
{
    int xblks, yblks;
    int dims[2];
    PyArrayObject *desc;


    if (!PyArg_ParseTuple(args, "ii", &xblks, &yblks))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }

    if ( (xblks > MAX_BLOCKS) or (yblks > MAX_BLOCKS) ) {
        printf("error: block count too large, recompile for larger MAX_BLOCKS\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    
    dims[0] = DESCRIPTOR_SIZE(xblks, yblks);
    dims[1] = 1;

    if ( !(desc = (PyArrayObject*)PyArray_FromDims(2, dims, NPY_DOUBLE)) ) {
        printf("Error allocating Array\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    //return  PyArray_Return(desc);
    return Py_BuildValue("(Oii)", desc, xblks, yblks);
}

PyObject *Get_Descriptor(PyObject *obj, PyObject *args)
{
    int xblks, yblks, xoffset, win_width;
    PyArrayObject *desc;


    if (!PyArg_ParseTuple(args, "(O!ii)ii", &PyArray_Type, &desc, &xblks, &yblks, &xoffset, &win_width))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }

    Fill_Descriptor((double *)(desc->data), xoffset, win_width, xblks, yblks);

    Py_INCREF(Py_None);
    return Py_None;

}

//#############################################PYTHON INTERFACE###############################################################
extern "C" {
	static PyMethodDef libgistMethods[] = {
		{"init", Init_GIST, METH_VARARGS},
		{"process", Process_Image, METH_VARARGS},
        {"alloc", Descriptor_Allocate, METH_VARARGS},
        {"get", Get_Descriptor, METH_VARARGS},
        {"init_pca", Init_PCA, METH_VARARGS},
        {"pca_project", PCA_project, METH_VARARGS},

		/*{"GIST_ProcessT_Get_Info",  GIST_Get_Info, METH_VARARGS},
		{"GIST_PCA_new",    GIST_PCA_new, METH_VARARGS},
		{"GIST_Process",    GIST_Process, METH_VARARGS},
		{"GIST_Get_Descriptor_Alloc", GIST_Get_Descriptor_Alloc, METH_VARARGS},
		{"GIST_Get_Descriptor_Reuse", GIST_Get_Descriptor_Reuse, METH_VARARGS},
		{"GIST_Get_Descriptor_Rectangle_Alloc", GIST_Get_Descriptor_Rectangle_Alloc, METH_VARARGS},
		{"GIST_Get_Descriptor_Rectangle_Reuse", GIST_Get_Descriptor_Rectangle_Reuse, METH_VARARGS},
		{"GIST_Get_Descriptor_PCA_Reuse", GIST_Get_Descriptor_PCA_Reuse, METH_VARARGS},
		{"GIST_Get_Descriptor_PCA_Alloc", GIST_Get_Descriptor_PCA_Alloc, METH_VARARGS},*/
		{NULL, NULL}
	};

	void initlibgist()  {
		(void) Py_InitModule("libgist", libgistMethods);
		Py_Initialize();
		import_array();  // Must be present for NumPy.  Called first after above line.
	}
}