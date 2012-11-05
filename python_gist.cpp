
#include "python_gist.h"
#include <vector>
#include <opencv/cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>

#include <numpy/ndarrayobject.h>


int IMAGE_HEIGHT = 240, IMAGE_WIDTH = 320;
//std::vector<cv::Mat>       *Gabor_filters;
std::vector<cv::gpu::GpuMat> *Gabor_filters;
std::vector<cv::Mat>         *Response_Image;
std::vector<cv::Mat>         *Response_Image_test;

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

cv::Mat mkGaborKernel(int ks, float sig, float th, float lm, float ps)
{
    int hks = (ks-1)/2;
    float theta = th*CV_PI/180;
    float psi = ps*CV_PI/180;
    float del = 2.0/(ks-1);
    float lmbd =  0.5+lm/100.0;
    float sigma = sig/ks;
    float x_theta;
    float y_theta;
    cv::Mat kernel(ks,ks, CV_32F);
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}

std::vector<cv::gpu::GpuMat> *create_gabor(int nscales,  int *orientations)

{
    int nfilters = 0;
    int kernel_size = 21;
    
    for (int i=0; i < nscales; i++) nfilters += orientations[i];

    //std::vector<cv::Mat> *Gfs          = new std::vector<cv::Mat>(nfilters);
    std::vector<cv::gpu::GpuMat> *Gfs = new std::vector< cv::gpu::GpuMat >(nfilters);

    int filter = 0;
    for (int scale = 0; scale < nscales; scale++){
        for (int ori =0; ori < orientations[scale]; ori++) {
            printf("Allocate gabor gpu buffer %d\n", filter);
            (*Gfs)[filter] = cv::gpu::GpuMat(kernel_size, kernel_size, CV_32F);
            (*Gfs)[filter].upload(mkGaborKernel(kernel_size, 4 - scale, ori * 90/orientations[scale], 50, 90));
            filter++;
        }
    }
    
    
    return Gfs;
}

//=============================================================================================================================

std::vector<cv::Mat> *response_init(int width, int height)
{
	std::vector<cv::Mat> *response_images =  (std::vector<cv::Mat> *) new std::vector<cv::Mat>;
    
	for (int scl=0; scl < N_SCALES; scl++) {
		for (int ori=0; ori < orientations[scl]; ori++) {
			response_images->push_back(cv::Mat(height+1, width+1, CV_32FC1)); // +1 for some Integral image stuff later
            
		}
	}

	return response_images;
}
//=============================================================================================================================


cv::Mat tmp_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);

void format_image(cv::Mat &input, cv::Mat &output)
{
    
    cv::resize(input, output, cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT));
    
    cv::cvtColor(output, tmp_image, CV_BGR2GRAY);
    tmp_image.convertTo(output, CV_32FC1, 1.0/255.0);
    
    
}
//=============================================================================================================================
/*cv::Mat prefilt_process(cv::Mat &im, int fc)
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
            //((float*)im.data)[j*im.cols+i] = (float)log( ((float*)im.data)[j*im.cols+i]+1.0f );
            im.at<float>(j,i) = log(im.at<float>(j,i)+1.0);
        }
    }
    
    //
    // Add padding
    //copyMakeBorder(im, pim, 5, 5, 5, 5,  IPL_BORDER_REPLICATE);
    copyMakeBorder(im, pim, 5, 5, 5, 5, IPL_BORDER_CONSTANT, cv::Scalar(0,0,0));

    width  = pim.cols;
    height = pim.rows;

    //
    // Build whitening filter and apply whitening filter
    float s1 = fc/sqrt(log(2.0));
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            in1[j*width + i][0] = pim.at<float>(j,i);
            in1[j*width + i][1] = 0.0f;
            

            fx[j*width + i] = (float) i - width/2.0;
            fy[j*width + i] = (float) j - height/2.0;

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
            //((float *)pim.data)[j*pim.cols+i] -= in2[j*width+i][0] / (width*height);
            pim.at<float>(j,i) -= in2[j*width+i][0] / (width*height);

            in1[j*width + i][0] = pim.at<float>(j,i) * pim.at<float>(j,i);
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
            pim.at<float>(j,i) = pim.at<float>(j,i) / (0.2+sqrt(sqrt(in2[j*width+i][0]*in2[j*width+i][0]+in2[j*width+i][1]*in2[j*width+i][1]) / (width*height)));
        }
    }
    
    
    //
    // Remove borders
    cv::Mat res = pim(cv::Rect(5, 5, pim.cols-10, pim.rows-10));
    
    return res;
}*/
    
//=============================================================================================================================
void Process(cv::Mat &im)
{
    cv::gpu::GpuMat gpu_tmp(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F); // Might want to move this out to avoid any allocations that occur each time the function is called
    
    cv::gpu::GpuMat gpu_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F);
    
    //im = prefilt_process(im, 4);
    clock_t start = clock();

    //cv::gpu::GpuMat gpu_int_img;

    gpu_img.upload(im);

    for (unsigned int k=0; k < Gabor_filters->size(); k++) {
    
        //cv::gpu::filter2D(gpu_img, gpu_tmp, CV_32F, (*Gabor_filters)[k]);
        //printf("Convolve\n");
        cv::gpu::convolve(gpu_img, (*Gabor_filters)[k], gpu_tmp);
        //printf("power\n");
        cv::gpu::pow(gpu_tmp, 2.0, gpu_tmp);
        
    
        gpu_tmp.download(tmp_image);
        

        if (k == 7) {
            cv::normalize(tmp_image, tmp_image, 0, 1, CV_MINMAX);
            cv::imshow("Training Images", tmp_image);
            //cv::waitKey(0);
        }
    
        cv::integral(tmp_image, (*Response_Image)[k]);
    
    }

    clock_t end = clock();
    printf("Proc time %f\n", float(end - start)/CLOCKS_PER_SEC );
    
}
//=============================================================================================================================
#define MAX_BLOCKS 10
#define DESCRIPTOR_SIZE(x, y) x * y * Gabor_filters->size()

int nx[MAX_BLOCKS], ny[MAX_BLOCKS];


void Fill_Descriptor(float *desc, 
                     int xoffset,   int yoffset, 
                     int win_width, int win_height,
                     int xblks,     int yblks)
{

    int i, x, y;
    
    int cols = IMAGE_WIDTH, rows = IMAGE_HEIGHT;

    int hlfimg_width   = cols/2;
    int hlfimg_height  = rows/2;

    int hlfwin_width = win_width/2;
    int hlfwin_height = win_height/2;

    int width  = (win_width/xblks)-1;
    int height = (win_height/yblks)-1;

    xoffset = xoffset + hlfimg_width  - hlfwin_width;
    yoffset = yoffset + hlfimg_height - hlfwin_height;

    


    if (xoffset < 0)                           xoffset = 0;
    if (xoffset > (IMAGE_WIDTH - win_width))   xoffset = IMAGE_WIDTH - win_width;

    if (yoffset < 0)                           yoffset = 0;
    if (yoffset > (IMAGE_HEIGHT - win_height)) yoffset = IMAGE_HEIGHT - win_height;

    nx[0] = xoffset;
    ny[0] = yoffset;
    
    
    
    for( i=1; i < xblks+1; i++) {
        nx[i] = (nx[i-1] + width);
    }
    
    for( i=1; i < yblks+1; i++) {
        ny[i] = (ny[i-1] + height);
    }
    

    float denom = (float)(ny[1]-ny[0])*(nx[1]-nx[0]);
   

    for (unsigned int gbr = 0; gbr < Response_Image->size(); gbr++) {

        cv::Mat src = (*Response_Image)[gbr];
        float *res = desc+gbr*xblks*yblks;

        for(y = 0; y < yblks; y++) {
            for(x = 0; x < xblks; x++) {  
                float mean   = (  src.at<float>(ny[y+1], nx[x+1]) 
                                 + src.at<float>(ny[y], nx[x]) 
                                 - src.at<float>(ny[y+1], nx[x])
                                 - src.at<float>(ny[y], nx[x+1]) );
                

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

    Get_cvMat_From_Numpy_Mat(desc, cvdesc, CV_32FC1);
    Get_cvMat_From_Numpy_Mat(pca_desc, cvpca_desc, CV_32FC1);

    /*printf("%d %d : %d %d data %d %d : %d %d\n", cvdesc.cols, cvdesc.rows, cvpca_desc.cols, cvpca_desc.rows,   
                                                 pca_object.mean.cols, pca_object.mean.rows, pca_object.eigenstd::vectors.cols, pca_object.eigenstd::vectors.rows);
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
	

	Gabor_filters = create_gabor(N_SCALES, orientations);
	Response_Image = response_init(cols, rows);

	IMAGE_WIDTH  = cols;
	IMAGE_HEIGHT = rows;

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject *Cleanup_GIST(PyObject *obj, PyObject *args)
{


    Py_INCREF(Py_None);
    return Py_None;
}

/*
PyObject *Init_PCA(PyObject* obj, PyObject *args)
{
    PyArrayObject *mean, *eigenstd::vectors;
    

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &mean, &PyArray_Type, &eigenstd::vectors))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }
    
    Get_cvMat_From_Numpy_Mat(mean, pca_object.mean, CV_32FC1);
    Get_cvMat_From_Numpy_Mat(eigenstd::vectors, pca_object.eigenstd::vectors, CV_32FC1);
    
    PCcount = pca_object.eigenstd::vectors.rows;

    
    return Py_BuildValue("(ii)", PCcount, pca_object.mean.cols);

}
*/

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

    
    dims[0] = 1;
    dims[1] = DESCRIPTOR_SIZE(xblks, yblks);

    if ( !(desc = (PyArrayObject*)PyArray_FromDims(2, dims, NPY_FLOAT)) ) {
        printf("Error allocating Array\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    //return  PyArray_Return(desc);
    return Py_BuildValue("(Oii)", desc, xblks, yblks);
}

PyObject *Get_Descriptor(PyObject *obj, PyObject *args)
{
    int xblks, yblks, xoffset, win_width, yoffset, win_height;
    PyArrayObject *desc;


    if (!PyArg_ParseTuple(args, "(O!ii)iiii", &PyArray_Type, &desc, &xblks, &yblks, &xoffset, &yoffset, &win_width, &win_height))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }

    Fill_Descriptor((float *)(desc->data), xoffset, yoffset, win_width, win_height, xblks, yblks);

    Py_INCREF(Py_None);
    return Py_None;

}

//#############################################PYTHON INTERFACE###############################################################
extern "C" {
	static PyMethodDef libgistMethods[] = {
		{"init", Init_GIST, METH_VARARGS},
        {"cleanup", Cleanup_GIST, METH_VARARGS},
		{"process", Process_Image, METH_VARARGS},
        {"alloc", Descriptor_Allocate, METH_VARARGS},
        {"get", Get_Descriptor, METH_VARARGS},
        //{"init_pca", Init_PCA, METH_VARARGS},
       // {"pca_project", PCA_project, METH_VARARGS},

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