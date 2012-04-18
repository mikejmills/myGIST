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


PyObject* GIST_basic_new(PyObject* obj, PyObject*args)
{
	
	PyArrayObject *baseim;
	int blocks;
	cv::Mat output;
	
	cv::namedWindow("test",1);

	if (!PyArg_ParseTuple(args, "O!i",  &PyArray_Type,  &baseim, &blocks))  return NULL;
	Get_Mat(baseim);
	format_image(tmp, output);
    Gist_Processor *gp = new Gist_Processor(output, blocks);
    
    return Py_BuildValue("l", (long int)gp);
}

PyObject* GIST_PCA_new(PyObject* obj, PyObject*args)
{
	
	PyArrayObject *baseim, *blocks;
	
	cv::Mat output;
	
	cv::namedWindow("test",1);

	if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type,  &baseim, &PyArray_Type,  &blocks))  return NULL;

	Get_Mat(baseim);
	format_image(tmp, output);
    
    Gist_Processor *gp = new Gist_Processor(output, (long int *)(blocks->data), (int)blocks->dimensions[1]);
    
    return Py_BuildValue("l", (long int)gp);
}

PyObject* GIST_Process(PyObject* obj, PyObject*args)
{
	PyArrayObject *imarray;
	Gist_Processor *gp;

	if (!PyArg_ParseTuple(args, "O!l",  &PyArray_Type,  &imarray,  ((long int) &gp)))  return NULL;

	Get_Mat(imarray);
	cv::Mat output;
	
	format_image(tmp, output);
	gp->Process(output);
	
	return Py_None;
}

PyObject* GIST_Get_Descriptor_Alloc(PyObject* obj, PyObject*args)
{
	Gist_Processor *gp;
	PyArrayObject  *pydesc;
	int blocks, x, y;
	double *desc;

	int dims[1];
	
	if (!PyArg_ParseTuple(args, "iiil", &blocks, &x, &y, ((long int) &gp)) )  return NULL;
	
	int size = gp->Get_Descriptor(&desc, blocks, x, y);
	dims[0] = size;

	pydesc = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_DOUBLE);


	for (int i=0; i < size; i++) {
		((double *)(pydesc->data))[i] = desc[i];
	}
	
	gist_free(desc);
	
	return  PyArray_Return(pydesc);
}

PyObject* GIST_Get_Descriptor_Reuse(PyObject* obj, PyObject*args)
{
	Gist_Processor *gp;
	PyArrayObject  *pydesc;

	int blocks, x, y;
	
	if (!PyArg_ParseTuple(args, "O!iiil", &PyArray_Type, &pydesc, &blocks, &x, &y, ((long int) &gp)) )  return NULL;
	
	gp->Get_Descriptor(((double *)(pydesc->data)), blocks, x, y);
	
	return  Py_None; //PyArray_Return(pydesc);
}



extern "C" {
	static PyMethodDef libgistMethods[] = {
		{"GIST_basic_new",  GIST_basic_new, METH_VARARGS},
		{"GIST_PCA_new",    GIST_PCA_new, METH_VARARGS},
		{"GIST_Process",    GIST_Process, METH_VARARGS},
		{"GIST_Get_Descriptor_Alloc", GIST_Get_Descriptor_Alloc, METH_VARARGS},
		{"GIST_Get_Descriptor_Reuse", GIST_Get_Descriptor_Reuse, METH_VARARGS},
		/*{"GIST_Get_Descriptor_PCA_Reuse", GIST_Get_Descriptor_PCA_Reuse, METH_VARARGS},
		{"GIST_Get_Descriptor_PCA_Alloc", GIST_Get_Descriptor_PCA_Alloc, METH_VARARGS},*/
		{NULL, NULL}
	};

	void initlibgist()  {
		(void) Py_InitModule("libgist", libgistMethods);
		import_array();  // Must be present for NumPy.  Called first after above line.
	}
}