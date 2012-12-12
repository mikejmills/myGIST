
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

int IMAGE_HEIGHT = 480, IMAGE_WIDTH = 640;
//std::vector<cv::Mat>       *Gabor_filters;
std::vector<cv::gpu::GpuMat> *Gabor_filters;
std::vector<cv::gpu::GpuMat> *Gpu_Response_Images;
std::vector<cv::Mat>         *Response_Image;
std::vector<cv::Mat>         *Response_Image_test;



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
            x_theta = x*del*cosf(theta)+y*del*sinf(theta);
            y_theta = -x*del*sinf(theta)+y*del*cosf(theta);
            kernel.at<float>(hks+y,hks+x) = (float)expf(-0.5*(powf(x_theta,2)+powf(y_theta,2))/powf(sigma,2))* cosf(2*CV_PI*x_theta/lmbd + psi);
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
            (*Gfs)[filter] = cv::gpu::GpuMat(KERNEL_SIZE, KERNEL_SIZE, CV_32F);
            (*Gfs)[filter].upload(mkGaborKernel(KERNEL_SIZE, 4 - scale, ori * 90/orientations[scale], 50, 90));
            
            printf("filter %d %d %d\n", filter, 4 - scale, ori*90/orientations[scale]);
            
            filter++;
        }
    }
    
    
    return Gfs;
}

//=============================================================================================================================

std::vector<cv::gpu::GpuMat> *create_gpu_response_images(int nfilters, int width, int height)

{
    
    std::vector<cv::gpu::GpuMat> *gpu_responses = new std::vector< cv::gpu::GpuMat >(nfilters);

    for (int f = 0; f < nfilters; f++){
        (*gpu_responses)[f] = cv::gpu::GpuMat(height, width, CV_32F);
    }
    
    return gpu_responses;
}


//=============================================================================================================================
cv::Mat ResponseDebugImage;

std::vector<cv::Mat> *response_init(int width, int height)
{
	std::vector<cv::Mat> *response_images =  (std::vector<cv::Mat> *) new std::vector<cv::Mat>;
    


	for (int scl=0; scl < N_SCALES; scl++) {
		for (int ori=0; ori < orientations[scl]; ori++) {
			response_images->push_back(cv::Mat(height, width, CV_64F)); // +1 for some Integral image stuff later
                
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
/*
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
    


cv::gpu::GpuMat gpu_img, gpu_tmp, gpu_integral;
cv::Mat         proc_img;

void prefilt_process()
{
    cv::gpu::GpuMat gpu_img_blur, lowContrastMask, sharpened;
    double amount = 1;

    cv::gpu::GaussianBlur(gpu_img, gpu_img_blur, cv::Size(9,9), 2.0, 2.0); // cv::BORDER_DEFAULT, -1);
    cv::gpu::subtract(gpu_img, gpu_img_blur, lowContrastMask);
    cv::gpu::abs(lowContrastMask, lowContrastMask);
    cv::gpu::threshold(lowContrastMask, lowContrastMask, 0.2, 1, cv::THRESH_BINARY);

    cv::gpu::multiply(gpu_img, 1+amount, sharpened);
    cv::gpu::multiply(gpu_img_blur, -amount, gpu_img_blur);
    cv::gpu::add(sharpened, gpu_img_blur, gpu_img);

    //cv::gpu::GpuMat lowContrastMask = abs(gpu_img - gpu_img_blur) < 0.5;
    //cv::gpu::GpuMat sharpened = gpu_img*(1+amount) + gpu_img_blur*(-amount);
    //img.copyTo(sharpened, gpu_img);
}

//=============================================================================================================================
void Process(cv::Mat &im)
{
    //clock_t start = clock();
    //im = prefilt_process(im, 4);

    im.convertTo(im, CV_32F);
    
    //cv::imshow("Response Image", im);
    gpu_img.upload(im);
    
    //prefilt_process();
    //gpu_img.download(im);

    //
    // Do the GPU calculations and store the results into several Response images
    for (unsigned int k=0; k < Gabor_filters->size(); k++) {
        cv::gpu::convolve(gpu_img, (*Gabor_filters)[k], gpu_tmp);
        cv::gpu::pow(gpu_tmp, 2.0, (*Gpu_Response_Images)[k]);
    }

    //clock_t start = clock();
    //
    // Download and integrate all the results into the host memory
    
    
    
    for (unsigned int k=0; k < Gabor_filters->size(); k++) {
        (*Gpu_Response_Images)[k].download(proc_img); 
        
        if (k == 9) {
            cv::imshow("Response Image", proc_img);
        }

        cv::integral(proc_img, (*Response_Image)[k]);
        
    }

    //cv::imshow("Response Image", proc_img);      
    
    //clock_t end = clock();
    //printf("Download Time %f\n", float(end - start)/CLOCKS_PER_SEC );
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
    
    // 
    // This is required because of the convolution
    int cols = IMAGE_WIDTH - (KERNEL_SIZE + 1), rows = IMAGE_HEIGHT- (KERNEL_SIZE + 1);

    //
    // make sure we are not asking for a window greater then the image after convolution
    if (win_width > cols)  win_width  = cols;
    if (win_height > rows) win_height = rows;


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

                double mean   = (  src.at<double>(ny[y+1], nx[x+1]) 
                                 + src.at<double>(ny[y], nx[x]) 
                                 - src.at<double>(ny[y+1], nx[x])
                                 - src.at<double>(ny[y], nx[x+1]) );
                

                res[y*xblks+x]=  mean / denom;
               
             }
        }
    
    }


}

PyArrayObject *Allocate_Descriptor_Mem(int xblks, int yblks)
{
    int dims[2];
    PyArrayObject *desc;



    if ( (xblks > MAX_BLOCKS) or (yblks > MAX_BLOCKS) ) {
        printf("error: block count too large, recompile for larger MAX_BLOCKS\n");
        return NULL;
    }

    
    dims[0] = 1;
    dims[1] = DESCRIPTOR_SIZE(xblks, yblks);

    if ( !(desc = (PyArrayObject*)PyArray_FromDims(2, dims, NPY_DOUBLE)) ) {
        printf("Error allocating Array\n");
        
        return NULL;
    }

    return desc;
}



cv::gpu::FAST_GPU Gpu_Fast_Corner_Detector(50, TRUE); // Look into changing the threshold
cv::gpu::GpuMat   FAST_gpu_im;
    
std::vector<cv::KeyPoint> Get_Corners(cv::Mat im, int n_points)
{
    std::vector<cv::KeyPoint> keypoints, ret_points(n_points);
    srand ( time(NULL) );

    im.convertTo(im, CV_8UC1, 255);
    
    FAST_gpu_im.upload(im);
    
    Gpu_Fast_Corner_Detector(FAST_gpu_im, cv::gpu::GpuMat(), keypoints);
    
    if (keypoints.size() <= 0) {
        return ret_points;
    } 

    for (int i=0; i < n_points; i++) {
        ret_points[i] = keypoints[int(rand() % keypoints.size())];
    }

    return ret_points;
}

PyObject *Create_Keypoint_Descriptor(cv::KeyPoint pt, int w, int h, int xblks, int yblks, PyArrayObject *desc)
{
    
    

    //Fill_Descriptor( (double *)(desc->data) + offset, pt.pt.x, pt.pt.y, w, h, xblks, yblks);

    //
    // ( (x,y), (w,h), (xblks, yblks))
    return Py_BuildValue("( (ii), (ii), (ii), O)", pt.pt.x, pt.pt.y, w, h, xblks, yblks, desc);
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

    tmp_image      = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_64F);
    proc_img       = cv::Mat(IMAGE_HEIGHT-(KERNEL_SIZE+1), IMAGE_WIDTH-(KERNEL_SIZE+1), CV_32F);
    printf("Proc Img Initialization %d %d\n", IMAGE_WIDTH-(KERNEL_SIZE+1), IMAGE_HEIGHT-(KERNEL_SIZE+1));
    gpu_img        = cv::gpu::GpuMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F);
    gpu_tmp        = cv::gpu::GpuMat(IMAGE_HEIGHT-(KERNEL_SIZE+1), IMAGE_WIDTH-(KERNEL_SIZE+1), CV_32F);
    gpu_integral   = cv::gpu::GpuMat(IMAGE_HEIGHT-(KERNEL_SIZE+1)+1, IMAGE_WIDTH-(KERNEL_SIZE+1)+1, CV_32F);
    
    Response_Image      = response_init(IMAGE_WIDTH-(KERNEL_SIZE+1), IMAGE_HEIGHT-(KERNEL_SIZE+1));
    Gpu_Response_Images = create_gpu_response_images(Gabor_filters->size(), IMAGE_HEIGHT-(KERNEL_SIZE+1), IMAGE_WIDTH-(KERNEL_SIZE+1));



	Py_INCREF(Py_None);
	return Py_None;
}

PyObject *Cleanup_GIST(PyObject *obj, PyObject *args)
{

    Py_INCREF(Py_None);
    return Py_None;
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

    desc = Allocate_Descriptor_Mem(xblks, yblks);
    
    
    return Py_BuildValue("(Oii)", desc, xblks, yblks);
}

//
// Get the gist descriptor from the current processed image
//   Params: Desc, xblks, yblks, xoffset, yoffset, win_width, win_height
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

//
// Create a descriptor based on the locations of points
//   params image, num_points, width, height, xblks, yblks
PyObject *Create_Corner_Descriptor(PyObject *obj, PyObject *args)
{
    cv::Mat        output, tmp;
    PyArrayObject  *imarray;
    int win_width, win_height, xblks, yblks, n_points;

    if (!PyArg_ParseTuple(args, "O!iiiii",  &PyArray_Type,  &imarray, &n_points, &win_width, &win_height, &xblks, &yblks))  {
        printf("FAILED PROCESSING Parsing\n");
        return NULL;
    }

    Py_INCREF(imarray);
    
    Get_cvMat_From_Numpy_Mat(imarray, tmp, 16);
    format_image(tmp, output);
    
    std::vector<cv::KeyPoint> keypoints = Get_Corners(output, n_points);
    printf("returned keypoints size %d\n", keypoints.size());
    
    PyObject *desc_list = PyList_New(keypoints.size());

    for (int i=0; i < keypoints.size(); i++) {
        //PyObject *desc = Create_Keypoint_Descriptor(keypoints[i], win_width, win_height, xblks, yblks);
        //PyList.SetItem(desc_list, i, desc);
    }

    // Detect Fast Points return n
    // At each location define a descriptor (w, h) (xblks, yblks)
    // for each location
    //     get descriptor return a ((x_loc,y_loc), (w,h), (xblks, yblks), desc)

    // return the full structure structure 

    Py_DECREF(imarray);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Get_Corner_Descriptor(PyObject *obj, PyObject *args)
{
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
        {"create_corner_descriptor", Create_Corner_Descriptor, METH_VARARGS},
        {NULL, NULL}
	};

	void initlibgist()  {
		(void) Py_InitModule("libgist", libgistMethods);
		Py_Initialize();
		import_array();  // Must be present for NumPy.  Called first after above line.
	}
}