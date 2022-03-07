/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Python Wrapper
 * Title:        cmsismodule.h
 * Description:  C code for the CMSIS-DSP Python wrapper
 *
 * $Date:        27 April 2021
 * $Revision:    V1.0
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define MODNAME "cmsisdsp_matrix"
#define MODINITNAME cmsisdsp_matrix

#include "cmsisdsp_module.h"

MATRIXFROMNUMPY(f32,float32_t,double,NPY_DOUBLE);
MATRIXFROMNUMPY(f64,float64_t,double,NPY_DOUBLE);
MATRIXFROMNUMPY(q31,q31_t,int32_t,NPY_INT32);
MATRIXFROMNUMPY(q15,q15_t,int16_t,NPY_INT16);
MATRIXFROMNUMPY(q7,q7_t,int8_t,NPY_BYTE);

CREATEMATRIX(f32,float32_t);
CREATEMATRIX(f64,float64_t);
CREATEMATRIX(q31,q31_t);
CREATEMATRIX(q15,q15_t);
CREATEMATRIX(q7,q7_t);

NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);

NUMPYARRAYFROMMATRIX(f32,NPY_FLOAT);
NUMPYARRAYFROMMATRIX(f64,NPY_DOUBLE);
NUMPYARRAYFROMMATRIX(q31,NPY_INT32);
NUMPYARRAYFROMMATRIX(q15,NPY_INT16);
NUMPYARRAYFROMMATRIX(q7,NPY_BYTE);









typedef struct {
    PyObject_HEAD
    arm_matrix_instance_f32 *instance;
} dsp_arm_matrix_instance_f32Object;


static void
arm_matrix_instance_f32_dealloc(dsp_arm_matrix_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pData)
       {
          PyMem_Free(self->instance->pData);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_matrix_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_matrix_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_matrix_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_matrix_instance_f32));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_matrix_instance_f32_init(dsp_arm_matrix_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pData=NULL;
char *kwlist[] = {
"numRows","numCols","pData",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhO", kwlist,&self->instance->numRows
,&self->instance->numCols
,&pData
))
    {

    INITARRAYFIELD(pData,NPY_DOUBLE,double,float32_t);

    }
    return 0;
}

GETFIELD(arm_matrix_instance_f32,numRows,"h");
GETFIELD(arm_matrix_instance_f32,numCols,"h");


static PyMethodDef arm_matrix_instance_f32_methods[] = {

    {"numRows", (PyCFunction) Method_arm_matrix_instance_f32_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_matrix_instance_f32_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_matrix_instance_f32,arm_matrix_instance_f32_new,arm_matrix_instance_f32_dealloc,arm_matrix_instance_f32_init,arm_matrix_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_matrix_instance_f64 *instance;
} dsp_arm_matrix_instance_f64Object;


static void
arm_matrix_instance_f64_dealloc(dsp_arm_matrix_instance_f64Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pData)
       {
          PyMem_Free(self->instance->pData);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_matrix_instance_f64_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_matrix_instance_f64Object *self;
    //printf("New called\n");

    self = (dsp_arm_matrix_instance_f64Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_matrix_instance_f64));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_matrix_instance_f64_init(dsp_arm_matrix_instance_f64Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pData=NULL;
char *kwlist[] = {
"numRows","numCols","pData",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhO", kwlist,&self->instance->numRows
,&self->instance->numCols
,&pData
))
    {

    INITARRAYFIELD(pData,NPY_FLOAT64,float64_t,float64_t);

    }
    return 0;
}

GETFIELD(arm_matrix_instance_f64,numRows,"h");
GETFIELD(arm_matrix_instance_f64,numCols,"h");


static PyMethodDef arm_matrix_instance_f64_methods[] = {

    {"numRows", (PyCFunction) Method_arm_matrix_instance_f64_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_matrix_instance_f64_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_matrix_instance_f64,arm_matrix_instance_f64_new,arm_matrix_instance_f64_dealloc,arm_matrix_instance_f64_init,arm_matrix_instance_f64_methods);


typedef struct {
    PyObject_HEAD
    arm_matrix_instance_q15 *instance;
} dsp_arm_matrix_instance_q15Object;


static void
arm_matrix_instance_q15_dealloc(dsp_arm_matrix_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pData)
       {
          PyMem_Free(self->instance->pData);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_matrix_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_matrix_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_matrix_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_matrix_instance_q15));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_matrix_instance_q15_init(dsp_arm_matrix_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pData=NULL;
char *kwlist[] = {
"numRows","numCols","pData",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhO", kwlist,&self->instance->numRows
,&self->instance->numCols
,&pData
))
    {

    INITARRAYFIELD(pData,NPY_INT16,int16_t,int16_t);

    }
    return 0;
}

GETFIELD(arm_matrix_instance_q15,numRows,"h");
GETFIELD(arm_matrix_instance_q15,numCols,"h");


static PyMethodDef arm_matrix_instance_q15_methods[] = {

    {"numRows", (PyCFunction) Method_arm_matrix_instance_q15_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_matrix_instance_q15_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_matrix_instance_q15,arm_matrix_instance_q15_new,arm_matrix_instance_q15_dealloc,arm_matrix_instance_q15_init,arm_matrix_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_matrix_instance_q31 *instance;
} dsp_arm_matrix_instance_q31Object;


static void
arm_matrix_instance_q31_dealloc(dsp_arm_matrix_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pData)
       {
          PyMem_Free(self->instance->pData);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_matrix_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_matrix_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_matrix_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_matrix_instance_q31));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_matrix_instance_q31_init(dsp_arm_matrix_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pData=NULL;
char *kwlist[] = {
"numRows","numCols","pData",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhO", kwlist,&self->instance->numRows
,&self->instance->numCols
,&pData
))
    {

    INITARRAYFIELD(pData,NPY_INT32,int32_t,int32_t);

    }
    return 0;
}

GETFIELD(arm_matrix_instance_q31,numRows,"h");
GETFIELD(arm_matrix_instance_q31,numCols,"h");


static PyMethodDef arm_matrix_instance_q31_methods[] = {

    {"numRows", (PyCFunction) Method_arm_matrix_instance_q31_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_matrix_instance_q31_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_matrix_instance_q31,arm_matrix_instance_q31_new,arm_matrix_instance_q31_dealloc,arm_matrix_instance_q31_init,arm_matrix_instance_q31_methods);




void typeRegistration(PyObject *module) {

  
  ADDTYPE(arm_matrix_instance_f32);
  ADDTYPE(arm_matrix_instance_f64);
  ADDTYPE(arm_matrix_instance_q15);
  ADDTYPE(arm_matrix_instance_q31);

  
}






static PyObject *
cmsis_arm_mat_add_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f32 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f32 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f32 *pSrcA_converted = f32MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f32 *pSrcB_converted = f32MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_add_f32(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_add_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q15 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q15 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q15 *pSrcA_converted = q15MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q15 *pSrcB_converted = q15MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);

    arm_status returnValue = arm_mat_add_q15(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_add_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q31 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q31 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q31 *pSrcA_converted = q31MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q31 *pSrcB_converted = q31MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_add_q31(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_cmplx_trans_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_f32 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_f32 *pSrc_converted = f32MatrixFromNumpy(pSrc);
    pSrc_converted->numCols = pSrc_converted->numCols / 2;

    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows*2 ;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);
    /*
    Dimensions in matrix instance are not correct but they are not used
    by CMSIS-DSP since the library is not compiled with ARM_MATRIX_CHECK.
    So only source dimensions are used for the computation.

    The dimensions are correct for createf32Matrix since we need to create
    a bigger buffer and createf32Matrix only knows real.

    */

    arm_status returnValue = arm_mat_cmplx_trans_f32(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_cmplx_trans_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_q31 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_q31 *pSrc_converted = q31MatrixFromNumpy(pSrc);
    pSrc_converted->numCols = pSrc_converted->numCols / 2;

    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows*2 ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);
    /*
    Dimensions in matrix instance are not correct but they are not used
    by CMSIS-DSP since the library is not compiled with ARM_MATRIX_CHECK.
    So only source dimensions are used for the computation.

    The dimensions are correct for createf32Matrix since we need to create
    a bigger buffer and createf32Matrix only knows real.

    */

    arm_status returnValue = arm_mat_cmplx_trans_q31(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_cmplx_trans_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_q15 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_q15 *pSrc_converted = q15MatrixFromNumpy(pSrc);
    pSrc_converted->numCols = pSrc_converted->numCols / 2;

    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows*2 ;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);
    /*
    Dimensions in matrix instance are not correct but they are not used
    by CMSIS-DSP since the library is not compiled with ARM_MATRIX_CHECK.
    So only source dimensions are used for the computation.

    The dimensions are correct for createf32Matrix since we need to create
    a bigger buffer and createf32Matrix only knows real.

    */

    arm_status returnValue = arm_mat_cmplx_trans_q15(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_cmplx_mult_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f32 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f32 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f32 *pSrcA_converted = f32MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f32 *pSrcB_converted = f32MatrixFromNumpy(pSrcB);
    pSrcA_converted->numCols = pSrcA_converted->numCols / 2;
    pSrcB_converted->numCols = pSrcB_converted->numCols / 2;
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols * 2;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_cmplx_mult_f32(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_cmplx_mult_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q15 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q15 *pSrcB_converted=NULL; // input
  PyObject *pScratch=NULL; // input
  q15_t *pScratch_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OOO",&pSrcA,&pSrcB,&pScratch))
  {

    arm_matrix_instance_q15 *pSrcA_converted = q15MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q15 *pSrcB_converted = q15MatrixFromNumpy(pSrcB);
    GETARGUMENT(pScratch,NPY_INT16,int16_t,int16_t);
    pSrcA_converted->numCols = pSrcA_converted->numCols / 2;
    pSrcB_converted->numCols = pSrcB_converted->numCols / 2;
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols * 2;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);

    arm_status returnValue = arm_mat_cmplx_mult_q15(pSrcA_converted,pSrcB_converted,pDst_converted,pScratch_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_cmplx_mult_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q31 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q31 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q31 *pSrcA_converted = q31MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q31 *pSrcB_converted = q31MatrixFromNumpy(pSrcB);
    pSrcA_converted->numCols = pSrcA_converted->numCols / 2;
    pSrcB_converted->numCols = pSrcB_converted->numCols / 2;
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols * 2;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_cmplx_mult_q31(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_trans_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_f32 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_f32 *pSrc_converted = f32MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows ;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_trans_f32(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_trans_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_f64 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_f64 *pSrc_converted = f64MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows ;
    arm_matrix_instance_f64 *pDst_converted = createf64Matrix(row,column);

    arm_status returnValue = arm_mat_trans_f64(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf64Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_trans_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_q7 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_q7 *pSrc_converted = q7MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows ;
    arm_matrix_instance_q7 *pDst_converted = createq7Matrix(row,column);

    arm_status returnValue = arm_mat_trans_q7(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq7Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_trans_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_q15 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_q15 *pSrc_converted = q15MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows ;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);

    arm_status returnValue = arm_mat_trans_q15(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_trans_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_q31 *pSrc_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    arm_matrix_instance_q31 *pSrc_converted = q31MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numCols ;
    uint32_t column = pSrc_converted->numRows ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_trans_q31(pSrc_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_vec_mult_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f32 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  float32_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f32 *pSrcA_converted = f32MatrixFromNumpy(pSrcA);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcA_converted->numCols ;
    pDst=PyMem_Malloc(sizeof(float32_t)*row);

    arm_mat_vec_mult_f32(pSrcA_converted,pSrcB_converted,pDst);
    FLOATARRAY1(pDstOBJ,row,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_mult_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f32 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f32 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f32 *pSrcA_converted = f32MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f32 *pSrcB_converted = f32MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_mult_f32(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_mult_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f64 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f64 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f64 *pSrcA_converted = f64MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f64 *pSrcB_converted = f64MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_f64 *pDst_converted = createf64Matrix(row,column);

    arm_status returnValue = arm_mat_mult_f64(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf64Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_vec_mult_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q15 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  q15_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q15 *pSrcA_converted = q15MatrixFromNumpy(pSrcA);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,q15_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcA_converted->numCols ;
    pDst=PyMem_Malloc(sizeof(q15_t)*row);

    arm_mat_vec_mult_q15(pSrcA_converted,pSrcB_converted,pDst);
    INT16ARRAY1(pDstOBJ,row,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_vec_mult_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q7 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  q7_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q7 *pSrcA_converted = q7MatrixFromNumpy(pSrcA);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcA_converted->numCols ;
    pDst=PyMem_Malloc(sizeof(q7_t)*row);

    arm_mat_vec_mult_q7(pSrcA_converted,pSrcB_converted,pDst);
    INT8ARRAY1(pDstOBJ,row,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_mult_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q7 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q7 *pSrcB_converted=NULL; // input
  PyObject *pState=NULL; // input
  q7_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OOO",&pSrcA,&pSrcB,&pState))
  {

    arm_matrix_instance_q7 *pSrcA_converted = q7MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q7 *pSrcB_converted = q7MatrixFromNumpy(pSrcB);
    GETARGUMENT(pState,NPY_BYTE,int8_t,q7_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q7 *pDst_converted = createq7Matrix(row,column);

    arm_status returnValue = arm_mat_mult_q7(pSrcA_converted,pSrcB_converted,pDst_converted,pState_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq7Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pState_converted);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_mult_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q15 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q15 *pSrcB_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OOO",&pSrcA,&pSrcB,&pState))
  {

    arm_matrix_instance_q15 *pSrcA_converted = q15MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q15 *pSrcB_converted = q15MatrixFromNumpy(pSrcB);
    
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);

    arm_status returnValue = arm_mat_mult_q15(pSrcA_converted,pSrcB_converted,pDst_converted,pState_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pState_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_mult_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q15 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q15 *pSrcB_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OOO",&pSrcA,&pSrcB,&pState))
  {

    arm_matrix_instance_q15 *pSrcA_converted = q15MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q15 *pSrcB_converted = q15MatrixFromNumpy(pSrcB);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);

    arm_status returnValue = arm_mat_mult_fast_q15(pSrcA_converted,pSrcB_converted,pDst_converted,pState_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pState_converted);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_vec_mult_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q31 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  q31_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q31 *pSrcA_converted = q31MatrixFromNumpy(pSrcA);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,q31_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcA_converted->numCols ;
    pDst=PyMem_Malloc(sizeof(q31_t)*row);

    arm_mat_vec_mult_q31(pSrcA_converted,pSrcB_converted,pDst);
    INT32ARRAY1(pDstOBJ,row,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_mult_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q31 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q31 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q31 *pSrcA_converted = q31MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q31 *pSrcB_converted = q31MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_mult_q31(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_mult_opt_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q31 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q31 *pSrcB_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OOO",&pSrcA,&pSrcB,&pState))
  {

    arm_matrix_instance_q31 *pSrcA_converted = q31MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q31 *pSrcB_converted = q31MatrixFromNumpy(pSrcB);
    
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_mult_opt_q31(pSrcA_converted,pSrcB_converted,pDst_converted,pState_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pState_converted);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_mult_fast_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q31 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q31 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q31 *pSrcA_converted = q31MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q31 *pSrcB_converted = q31MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_mult_fast_q31(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_sub_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f32 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f32 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f32 *pSrcA_converted = f32MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f32 *pSrcB_converted = f32MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_sub_f32(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_sub_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f64 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f64 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f64 *pSrcA_converted = f64MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f64 *pSrcB_converted = f64MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_f64 *pDst_converted = createf64Matrix(row,column);

    arm_status returnValue = arm_mat_sub_f64(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf64Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_sub_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q15 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q15 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q15 *pSrcA_converted = q15MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q15 *pSrcB_converted = q15MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);

    arm_status returnValue = arm_mat_sub_q15(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_sub_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_q31 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_q31 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_q31 *pSrcA_converted = q31MatrixFromNumpy(pSrcA);
    arm_matrix_instance_q31 *pSrcB_converted = q31MatrixFromNumpy(pSrcB);
    uint32_t row = pSrcA_converted->numRows ;
    uint32_t column = pSrcB_converted->numCols ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_sub_q31(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_scale_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_f32 *pSrc_converted=NULL; // input
  float32_t scale; // input

  if (PyArg_ParseTuple(args,"Of",&pSrc,&scale))
  {

    arm_matrix_instance_f32 *pSrc_converted = f32MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numRows ;
    uint32_t column = pSrc_converted->numCols ;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_scale_f32(pSrc_converted,scale,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_scale_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_q15 *pSrc_converted=NULL; // input
  q15_t scaleFract; // input
  int32_t shift; // input

  if (PyArg_ParseTuple(args,"Ohi",&pSrc,&scaleFract,&shift))
  {

    arm_matrix_instance_q15 *pSrc_converted = q15MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numRows ;
    uint32_t column = pSrc_converted->numCols ;
    arm_matrix_instance_q15 *pDst_converted = createq15Matrix(row,column);

    arm_status returnValue = arm_mat_scale_q15(pSrc_converted,scaleFract,shift,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq15Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_scale_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  arm_matrix_instance_q31 *pSrc_converted=NULL; // input
  q31_t scaleFract; // input
  int32_t shift; // input

  if (PyArg_ParseTuple(args,"Oii",&pSrc,&scaleFract,&shift))
  {

    arm_matrix_instance_q31 *pSrc_converted = q31MatrixFromNumpy(pSrc);
    uint32_t row = pSrc_converted->numRows ;
    uint32_t column = pSrc_converted->numCols ;
    arm_matrix_instance_q31 *pDst_converted = createq31Matrix(row,column);

    arm_status returnValue = arm_mat_scale_q31(pSrc_converted,scaleFract,shift,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromq31Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_arm_mat_inverse_f32(PyObject *obj, PyObject *args)
{

  PyObject *src=NULL; // input
  arm_matrix_instance_f32 *src_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&src))
  {

    arm_matrix_instance_f32 *src_converted = f32MatrixFromNumpy(src);
    uint32_t row = src_converted->numCols ;
    uint32_t column = src_converted->numRows ;
    arm_matrix_instance_f32 *dst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_inverse_f32(src_converted,dst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* dstOBJ=NumpyArrayFromf32Matrix(dst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,dstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(src_converted);
    Py_DECREF(dstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_inverse_f64(PyObject *obj, PyObject *args)
{

  PyObject *src=NULL; // input
  arm_matrix_instance_f64 *src_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&src))
  {

    arm_matrix_instance_f64 *src_converted = f64MatrixFromNumpy(src);
    uint32_t row = src_converted->numCols ;
    uint32_t column = src_converted->numRows ;
    arm_matrix_instance_f64 *dst_converted = createf64Matrix(row,column);

    arm_status returnValue = arm_mat_inverse_f64(src_converted,dst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* dstOBJ=NumpyArrayFromf64Matrix(dst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,dstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(src_converted);
    Py_DECREF(dstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mat_cholesky_f32(PyObject *obj, PyObject *args)
{

  PyObject *src=NULL; // input
  arm_matrix_instance_f32 *src_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&src))
  {

    arm_matrix_instance_f32 *src_converted = f32MatrixFromNumpy(src);
    uint32_t column = src_converted->numCols ;
    uint32_t row = src_converted->numRows ;
    arm_matrix_instance_f32 *dst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_cholesky_f32(src_converted,dst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* dstOBJ=NumpyArrayFromf32Matrix(dst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,dstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(src_converted);
    Py_DECREF(dstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_cholesky_f64(PyObject *obj, PyObject *args)
{

  PyObject *src=NULL; // input
  arm_matrix_instance_f64 *src_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&src))
  {

    arm_matrix_instance_f64 *src_converted = f64MatrixFromNumpy(src);
    uint32_t column = src_converted->numCols ;
    uint32_t row = src_converted->numRows ;
    arm_matrix_instance_f64 *dst_converted = createf64Matrix(row,column);

    arm_status returnValue = arm_mat_cholesky_f64(src_converted,dst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* dstOBJ=NumpyArrayFromf64Matrix(dst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,dstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(src_converted);
    Py_DECREF(dstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_ldlt_f32(PyObject *obj, PyObject *args)
{

  PyObject *src=NULL; // input
  arm_matrix_instance_f32 *src_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&src))
  {

    arm_matrix_instance_f32 *src_converted = f32MatrixFromNumpy(src);
    uint32_t column = src_converted->numCols ;
    uint32_t row = src_converted->numRows ;
    
    arm_matrix_instance_f32 *l_converted = createf32Matrix(row,column);
    arm_matrix_instance_f32 *d_converted = createf32Matrix(row,column);

    uint16_t *pPerm=(uint16_t *)PyMem_Malloc(sizeof(uint16_t)*row);
    INT16ARRAY1(pPermOBJ,row,pPerm);

    arm_status returnValue = arm_mat_ldlt_f32(src_converted,l_converted,d_converted,pPerm);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    
    PyObject* lOBJ=NumpyArrayFromf32Matrix(l_converted);
    PyObject* dOBJ=NumpyArrayFromf32Matrix(d_converted);


    PyObject *pythonResult = Py_BuildValue("OOOO",theReturnOBJ,lOBJ,dOBJ,pPermOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(src_converted);
    Py_DECREF(lOBJ);
    Py_DECREF(dOBJ);
    Py_DECREF(pPermOBJ);

    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_ldlt_f64(PyObject *obj, PyObject *args)
{

  PyObject *src=NULL; // input
  arm_matrix_instance_f64 *src_converted=NULL; // input

  if (PyArg_ParseTuple(args,"O",&src))
  {

    arm_matrix_instance_f64 *src_converted = f64MatrixFromNumpy(src);
    uint32_t column = src_converted->numCols ;
    uint32_t row = src_converted->numRows ;
    
    arm_matrix_instance_f64 *l_converted = createf64Matrix(row,column);
    arm_matrix_instance_f64 *d_converted = createf64Matrix(row,column);

    uint16_t *pPerm=(uint16_t *)PyMem_Malloc(sizeof(uint16_t)*row);
    INT16ARRAY1(pPermOBJ,row,pPerm);

    arm_status returnValue = arm_mat_ldlt_f64(src_converted,l_converted,d_converted,pPerm);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    
    PyObject* lOBJ=NumpyArrayFromf64Matrix(l_converted);
    PyObject* dOBJ=NumpyArrayFromf64Matrix(d_converted);


    PyObject *pythonResult = Py_BuildValue("OOOO",theReturnOBJ,lOBJ,dOBJ,pPermOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(src_converted);
    Py_DECREF(lOBJ);
    Py_DECREF(dOBJ);
    Py_DECREF(pPermOBJ);

    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_solve_lower_triangular_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f32 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f32 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f32 *pSrcA_converted = f32MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f32 *pSrcB_converted = f32MatrixFromNumpy(pSrcB);
    uint32_t column = pSrcB_converted->numCols ;
    uint32_t row = pSrcA_converted->numRows ;

    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_solve_lower_triangular_f32(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_solve_lower_triangular_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f64 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f64 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f64 *pSrcA_converted = f64MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f64 *pSrcB_converted = f64MatrixFromNumpy(pSrcB);
    uint32_t column = pSrcB_converted->numCols ;
    uint32_t row = pSrcA_converted->numRows ;

    arm_matrix_instance_f64 *pDst_converted = createf64Matrix(row,column);

    arm_status returnValue = arm_mat_solve_lower_triangular_f64(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf64Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_solve_upper_triangular_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f32 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f32 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f32 *pSrcA_converted = f32MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f32 *pSrcB_converted = f32MatrixFromNumpy(pSrcB);
    uint32_t column = pSrcB_converted->numCols ;
    uint32_t row = pSrcA_converted->numRows ;
    arm_matrix_instance_f32 *pDst_converted = createf32Matrix(row,column);

    arm_status returnValue = arm_mat_solve_upper_triangular_f32(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf32Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mat_solve_upper_triangular_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  arm_matrix_instance_f64 *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  arm_matrix_instance_f64 *pSrcB_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    arm_matrix_instance_f64 *pSrcA_converted = f64MatrixFromNumpy(pSrcA);
    arm_matrix_instance_f64 *pSrcB_converted = f64MatrixFromNumpy(pSrcB);
    uint32_t column = pSrcB_converted->numCols ;
    uint32_t row = pSrcA_converted->numRows ;
    arm_matrix_instance_f64 *pDst_converted = createf64Matrix(row,column);

    arm_status returnValue = arm_mat_solve_upper_triangular_f64(pSrcA_converted,pSrcB_converted,pDst_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pDstOBJ=NumpyArrayFromf64Matrix(pDst_converted);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyMethodDef CMSISDSPMethods[] = {

{"arm_mat_add_f32",  cmsis_arm_mat_add_f32, METH_VARARGS,""},
{"arm_mat_add_q15",  cmsis_arm_mat_add_q15, METH_VARARGS,""},
{"arm_mat_add_q31",  cmsis_arm_mat_add_q31, METH_VARARGS,""},
{"arm_mat_cmplx_mult_f32",  cmsis_arm_mat_cmplx_mult_f32, METH_VARARGS,""},
{"arm_mat_cmplx_mult_q15",  cmsis_arm_mat_cmplx_mult_q15, METH_VARARGS,""},
{"arm_mat_cmplx_mult_q31",  cmsis_arm_mat_cmplx_mult_q31, METH_VARARGS,""},
{"arm_mat_trans_f32",  cmsis_arm_mat_trans_f32, METH_VARARGS,""},
{"arm_mat_trans_f64",  cmsis_arm_mat_trans_f64, METH_VARARGS,""},
{"arm_mat_trans_q15",  cmsis_arm_mat_trans_q15, METH_VARARGS,""},
{"arm_mat_trans_q31",  cmsis_arm_mat_trans_q31, METH_VARARGS,""},
{"arm_mat_trans_q7",  cmsis_arm_mat_trans_q7, METH_VARARGS,""},


{"arm_mat_vec_mult_f32",  cmsis_arm_mat_vec_mult_f32, METH_VARARGS,""},
{"arm_mat_vec_mult_q31",  cmsis_arm_mat_vec_mult_q31, METH_VARARGS,""},
{"arm_mat_vec_mult_q15",  cmsis_arm_mat_vec_mult_q15, METH_VARARGS,""},
{"arm_mat_vec_mult_q7",  cmsis_arm_mat_vec_mult_q7, METH_VARARGS,""},

{"arm_mat_mult_f32",  cmsis_arm_mat_mult_f32, METH_VARARGS,""},
{"arm_mat_mult_f64",  cmsis_arm_mat_mult_f32, METH_VARARGS,""},
{"arm_mat_mult_q7",  cmsis_arm_mat_mult_q7, METH_VARARGS,""},
{"arm_mat_mult_q15",  cmsis_arm_mat_mult_q15, METH_VARARGS,""},
{"arm_mat_mult_fast_q15",  cmsis_arm_mat_mult_fast_q15, METH_VARARGS,""},
{"arm_mat_mult_q31",  cmsis_arm_mat_mult_q31, METH_VARARGS,""},
{"arm_mat_mult_opt_q31",  cmsis_arm_mat_mult_opt_q31, METH_VARARGS,""},
{"arm_mat_mult_fast_q31",  cmsis_arm_mat_mult_fast_q31, METH_VARARGS,""},
{"arm_mat_sub_f32",  cmsis_arm_mat_sub_f32, METH_VARARGS,""},
{"arm_mat_sub_f64",  cmsis_arm_mat_sub_f64, METH_VARARGS,""},
{"arm_mat_sub_q15",  cmsis_arm_mat_sub_q15, METH_VARARGS,""},
{"arm_mat_sub_q31",  cmsis_arm_mat_sub_q31, METH_VARARGS,""},
{"arm_mat_scale_f32",  cmsis_arm_mat_scale_f32, METH_VARARGS,""},
{"arm_mat_scale_q15",  cmsis_arm_mat_scale_q15, METH_VARARGS,""},
{"arm_mat_scale_q31",  cmsis_arm_mat_scale_q31, METH_VARARGS,""},
{"arm_mat_inverse_f32",  cmsis_arm_mat_inverse_f32, METH_VARARGS,""},
{"arm_mat_inverse_f64",  cmsis_arm_mat_inverse_f64, METH_VARARGS,""},
{"arm_mat_cmplx_trans_f32",  cmsis_arm_mat_cmplx_trans_f32, METH_VARARGS,""},
{"arm_mat_cmplx_trans_q31",  cmsis_arm_mat_cmplx_trans_q31, METH_VARARGS,""},
{"arm_mat_cmplx_trans_q15",  cmsis_arm_mat_cmplx_trans_q15, METH_VARARGS,""},
{"arm_mat_cholesky_f32",  cmsis_arm_mat_cholesky_f32, METH_VARARGS,""},
{"arm_mat_cholesky_f64",  cmsis_arm_mat_cholesky_f64, METH_VARARGS,""},
{"arm_mat_ldlt_f32",  cmsis_arm_mat_ldlt_f32, METH_VARARGS,""},
{"arm_mat_ldlt_f64",  cmsis_arm_mat_ldlt_f64, METH_VARARGS,""},
{"arm_mat_solve_lower_triangular_f32",  cmsis_arm_mat_solve_lower_triangular_f32, METH_VARARGS,""},
{"arm_mat_solve_lower_triangular_f64",  cmsis_arm_mat_solve_lower_triangular_f64, METH_VARARGS,""},
{"arm_mat_solve_upper_triangular_f32",  cmsis_arm_mat_solve_upper_triangular_f32, METH_VARARGS,""},
{"arm_mat_solve_upper_triangular_f64",  cmsis_arm_mat_solve_upper_triangular_f64, METH_VARARGS,""},
{"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#ifdef IS_PY3K
static int cmsisdsp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int cmsisdsp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        MODNAME,
        NULL,
        sizeof(struct module_state),
        CMSISDSPMethods,
        NULL,
        cmsisdsp_traverse,
        cmsisdsp_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
CAT(PyInit_,MODINITNAME)(void)


#else
#define INITERROR return

void CAT(init,MODINITNAME)(void)
#endif
{
    import_array();

  #ifdef IS_PY3K
    PyObject *module = PyModule_Create(&moduledef);
  #else
    PyObject *module = Py_InitModule(MODNAME, CMSISDSPMethods);
  #endif

  if (module == NULL)
      INITERROR;
  struct module_state *st = GETSTATE(module);
  
  st->error = PyErr_NewException(MODNAME".Error", NULL, NULL);
  if (st->error == NULL) {
      Py_DECREF(module);
      INITERROR;
  }


  typeRegistration(module);

  #ifdef IS_PY3K
    return module;
  #endif
}