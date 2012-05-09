#include <iostream>
#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include "gist.h"

#define Get_Mat(matin) \
		const npy_intp* _strides = PyArray_STRIDES(matin);\
        const npy_intp* _sizes = PyArray_DIMS(matin);\
        int size[CV_MAX_DIM+1];\
        size_t step[CV_MAX_DIM+1];\
        size[0] = (int)_sizes[0];\
        size[1] = (int)_sizes[1];\
		step[0] = (size_t) _strides[0];\
        step[1] = (size_t) _strides[1];\
        cv::Mat tmp(2, size, 16, PyArray_DATA(matin), step);

cv::Mat tmp(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, NULL);
//long int blocks[] = { 4, 8, 10};
//Gist_Processor gp(tmp, blocks, 3);
Gist_Processor gp(tmp, 4);
/*
PyObject* GIST_basic_new(PyObject* obj, PyObject*args)
{
	
	PyArrayObject *baseim;
	int     blocks;
	cv::Mat output;
	

	if (!PyArg_ParseTuple(args, "O!i",  &PyArray_Type, &baseim, &blocks))  return NULL;
	Py_INCREF(baseim);
	
	Get_Mat(baseim);
	format_image(tmp, output);
    
    //Gist_Processor *gp = new Gist_Processor(output, blocks);

    //printf("Creation %d %d %d %d\n", sizeof(Gist_Processor *), sizeof(long), sizeof(long int), sizeof(long long int));
	Py_DECREF(baseim);
	
    return Py_BuildValue("k", (unsigned long int)100);
}

PyObject* GIST_PCA_new(PyObject* obj, PyObject*args)
{
	
	PyArrayObject *baseim, *blocks;
	
	cv::Mat output;

	if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type,  &baseim, &PyArray_Type,  &blocks))  return NULL;
	Py_INCREF(baseim);
	Py_INCREF(blocks);
	
	Get_Mat(baseim);
	format_image(tmp, output);

    //Gist_Processor *gp = new Gist_Processor(output, (long int *)(blocks->data), (int)blocks->dimensions[1]);
	
	Py_DECREF(blocks);
    Py_DECREF(baseim);
	    
    return Py_BuildValue("k", (unsigned long int)100);
}
*/

PyObject* GIST_Process(PyObject* obj, PyObject*args)
{
	cv::Mat        output;
	PyArrayObject  *imarray;

	if (!PyArg_ParseTuple(args, "O!",  &PyArray_Type,  &imarray))  {
		printf("FAILED PROCESSING Parsing\n");
		return NULL;
	}

	Py_INCREF(imarray);

	Get_Mat(imarray);
	format_image(tmp, output);
	gp.Process(output);
	
	Py_DECREF(imarray);
	

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* GIST_Get_Descriptor_Alloc(PyObject* obj, PyObject*args)
{
	
	PyArrayObject  *pydesc;
	//PyObject *pydesc;

	int blocks, x, y;
	double *desc;

	int dims[2];
	
	if (!PyArg_ParseTuple(args, "iii", &blocks, &x, &y )) {
		printf("Failed Descriptor Alloc Parse\n");
		return NULL;
	}
	
	int size = gp.Get_Descriptor(&desc, blocks, x, y);

	
	dims[0] = size;
	dims[1] = 0;

	pydesc = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
	if (pydesc == NULL) printf("Error allocating Array\n");
	
	Py_INCREF(pydesc);
        	
	
	memmove((void *)pydesc->data, (void *)desc, sizeof(double) * dims[0]);

	gist_free(desc);
	Py_DECREF(pydesc);

	return  PyArray_Return(pydesc);
	
}

PyObject* GIST_Get_Descriptor_Reuse(PyObject* obj, PyObject*args)
{
	PyArrayObject  *pydesc;

	int blocks, x, y;
	
	if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &pydesc, &blocks, &x, &y ))  return NULL;
	
	gp.Get_Descriptor((double *)(pydesc->data), blocks, x, y);
	
	Py_INCREF(Py_None);
	return Py_None;
}



extern "C" {
	static PyMethodDef libgistMethods[] = {
		/*{"GIST_basic_new",  GIST_basic_new, METH_VARARGS},
		{"GIST_PCA_new",    GIST_PCA_new, METH_VARARGS},*/
		{"GIST_Process",    GIST_Process, METH_VARARGS},
		{"GIST_Get_Descriptor_Alloc", GIST_Get_Descriptor_Alloc, METH_VARARGS},
		{"GIST_Get_Descriptor_Reuse", GIST_Get_Descriptor_Reuse, METH_VARARGS},
		/*{"GIST_Get_Descriptor_PCA_Reuse", GIST_Get_Descriptor_PCA_Reuse, METH_VARARGS},
		{"GIST_Get_Descriptor_PCA_Alloc", GIST_Get_Descriptor_PCA_Alloc, METH_VARARGS},*/
		{NULL, NULL}
	};

	void initlibgist()  {
		(void) Py_InitModule("libgist", libgistMethods);
		Py_Initialize();
		import_array();  // Must be present for NumPy.  Called first after above line.
	}
}