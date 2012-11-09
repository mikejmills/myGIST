
#include "python_gist.h"
#include <vector>
#include <cmath>
#include <opencv/cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>

#include <numpy/ndarrayobject.h>

#define KERNEL_SIZE 21

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

cv::Mat mkGaborKernel(int ks, double sig, double th, double lm, double ps)
{
    int hks = (ks-1)/2;
    double theta = th*CV_PI/180;
    double psi = ps*CV_PI/180;
    double del = 2.0/(ks-1);
    double lmbd =  0.5+lm/100.0;
    double sigma = sig/ks;
    double x_theta;
    double y_theta;
    cv::Mat kernel(ks,ks, CV_32F);
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<double>(hks+y,hks+x) = (double)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}

std::vector<cv::gpu::GpuMat> *create_gabor(int nscales,  int *orientations)

{
    int nfilters = 0;
        
    for (int i=0; i < nscales; i++) nfilters += orientations[i];

    std::vector<cv::gpu::GpuMat> *Gfs = new std::vector< cv::gpu::GpuMat >(nfilters);

    int filter = 0;
    for (int scale = 0; scale < nscales; scale++){
        for (int ori =0; ori < orientations[scale]; ori++) {
            (*Gfs)[filter] = cv::gpu::GpuMat(KERNEL_SIZE, KERNEL_SIZE, CV_64F);
            (*Gfs)[filter].upload(mkGaborKernel(KERNEL_SIZE, 4 - scale, ori * 90/orientations[scale], 50, 90));
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
			response_images->push_back(cv::Mat(height+1, width+1, CV_64F)); // +1 for some Integral image stuff later
            
		}
	}

	return response_images;
}
//=============================================================================================================================


cv::Mat tmp_image; //(IMAGE_HEIGHT, IMAGE_WIDTH, CV_64FC1);

void format_image(cv::Mat &input, cv::Mat &output)
{
    
    cv::resize(input, output, cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT));
    
    cv::cvtColor(output, tmp_image, CV_BGR2GRAY);
    tmp_image.convertTo(output, CV_64F, 1.0/255.0);
    
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
            //((double*)im.data)[j*im.cols+i] = (double)log( ((double*)im.data)[j*im.cols+i]+1.0f );
            im.at<double>(j,i) = log(im.at<double>(j,i)+1.0);
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
}*/
    
//    cv::gpu::GpuMat gpu_tmp(IMAGE_HEIGHT -22, IMAGE_WIDTH-22, CV_64F); // Might want to move this out to avoid any allocations that occur each time the function is called
//    cv::Mat proctmp(IMAGE_HEIGHT -22, IMAGE_WIDTH-22, CV_64F);
//    cv::gpu::GpuMat gpu_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_64F);

cv::gpu::GpuMat gpu_img, gpu_tmp;
cv::Mat         proc_img;

//=============================================================================================================================
void Process(cv::Mat &im)
{
    
    //im = prefilt_process(im, 4);
    clock_t start = clock();

    im.convertTo(im, CV_32F);
    
    gpu_img.upload(im);

    for (unsigned int k=0; k < Gabor_filters->size(); k++) {
    
        //cv::gpu::convolve(gpu_img, (*Gabor_filters)[k], gpu_tmp);
        //cv::gpu::pow(gpu_tmp, 2.0, gpu_tmp);
        
        //gpu_tmp.download(proc_img);
        
        gpu_tmp.download(im);

        //cv::normalize(proc_img, proc_img, 0, 1, CV_MINMAX);
        /*
        if (k == 0) {
            //cv::normalize(proc_img, proc_img, 0, 1, CV_MINMAX);
            cv::imshow("Training Images", proc_img);
            //cv::waitKey(0);
        }
            
        for (int y=0; y < proc_img.rows; y++)
            for (int x = 0; x < proc_img.cols; x++)
                if (std::isnan(proc_img.at<double>(y, x))) printf("*******PROC IMAGE FAIL*******\n"); 
        */
        
        cv::Mat tmp;
        //proc_img.convertTo(proc_im, CV_64F);
        im.convertTo(im, CV_64F);
        cv::integral(im, tmp);

        for (int y=0; y < tmp.rows; y++)
            for (int x = 0; x < tmp.cols; x++)
                if (std::isnan(tmp.at<double>(y, x))) printf("&&&&&&&&&INTEGRAL FAIL&&&&&&&&&\n"); 

    }

    clock_t end = clock();
    printf("Proc time %f\n", double(end - start)/CLOCKS_PER_SEC );
    
}
//=============================================================================================================================
#define MAX_BLOCKS 10
#define DESCRIPTOR_SIZE(x, y) x * y * Gabor_filters->size()

int nx[MAX_BLOCKS], ny[MAX_BLOCKS];


void Fill_Descriptor(double *desc, 
                     int xoffset,   int yoffset, 
                     int win_width, int win_height,
                     int xblks,     int yblks)
{

    int i, x, y;
    
    int cols = IMAGE_WIDTH - (KERNEL_SIZE + 1), rows = IMAGE_HEIGHT- (KERNEL_SIZE + 1);

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
    

    double denom = (double)(ny[1]-ny[0])*(nx[1]-nx[0]);
   

    for (unsigned int gbr = 0; gbr < Response_Image->size(); gbr++) {

        cv::Mat src = (*Response_Image)[gbr];
        
        double *res = desc+gbr*xblks*yblks;

        for(y = 0; y < yblks; y++) {
            for(x = 0; x < xblks; x++) { 

                double mean   = (   src.at<double>(ny[y+1], nx[x+1]) 
                                 + src.at<double>(ny[y], nx[x]) 
                                 - src.at<double>(ny[y+1], nx[x])
                                 - src.at<double>(ny[y], nx[x+1]) );
                

                res[y*xblks+x]=  mean / denom;
                //if (std::isnan(res[y*xblks+x])) printf("blah %f %f %f %f %f %f\n", mean, denom, src.at<double>(ny[y+1], nx[x+1]), src.at<double>(ny[y], nx[x]), src.at<double>(ny[y+1], nx[x]), src.at<double>(ny[y], nx[x+1]) );
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
	

	IMAGE_WIDTH  = cols;
	IMAGE_HEIGHT = rows;

    tmp_image = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F);
    proc_img  = cv::Mat(IMAGE_HEIGHT-(KERNEL_SIZE+1), IMAGE_WIDTH-(KERNEL_SIZE+1), CV_32F);
    gpu_img   = cv::gpu::GpuMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F);
    gpu_tmp   = cv::gpu::GpuMat(IMAGE_HEIGHT-(KERNEL_SIZE+1), IMAGE_WIDTH-(KERNEL_SIZE+1), CV_32F);
    
    Response_Image = response_init(IMAGE_HEIGHT-(KERNEL_SIZE+1), IMAGE_WIDTH-(KERNEL_SIZE+1));


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
    
    Get_cvMat_From_Numpy_Mat(mean, pca_object.mean, CV_64FC1);
    Get_cvMat_From_Numpy_Mat(eigenstd::vectors, pca_object.eigenstd::vectors, CV_64FC1);
    
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
    int xblks, yblks, xoffset, win_width, yoffset, win_height;
    PyArrayObject *desc;


    if (!PyArg_ParseTuple(args, "(O!ii)iiii", &PyArray_Type, &desc, &xblks, &yblks, &xoffset, &yoffset, &win_width, &win_height))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }

    Fill_Descriptor((double *)(desc->data), xoffset, yoffset, win_width, win_height, xblks, yblks);

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