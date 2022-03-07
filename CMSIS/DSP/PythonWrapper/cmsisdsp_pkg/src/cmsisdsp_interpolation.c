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

#define MODNAME "cmsisdsp_interpolation"
#define MODINITNAME cmsisdsp_interpolation

#include "cmsisdsp_module.h"


NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);


typedef struct {
    PyObject_HEAD
    arm_linear_interp_instance_f32 *instance;
} dsp_arm_linear_interp_instance_f32Object;


static void
arm_linear_interp_instance_f32_dealloc(dsp_arm_linear_interp_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pYData)
       {
          PyMem_Free(self->instance->pYData);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_linear_interp_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_linear_interp_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_linear_interp_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_linear_interp_instance_f32));

        self->instance->pYData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_linear_interp_instance_f32_init(dsp_arm_linear_interp_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pYData=NULL;
char *kwlist[] = {
"nValues","x1","xSpacing","pYData",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|iffO", kwlist,&self->instance->nValues
,&self->instance->x1
,&self->instance->xSpacing
,&pYData
))
    {

    INITARRAYFIELD(pYData,NPY_DOUBLE,double,float32_t);

    }
    return 0;
}

GETFIELD(arm_linear_interp_instance_f32,nValues,"i");
GETFIELD(arm_linear_interp_instance_f32,x1,"f");
GETFIELD(arm_linear_interp_instance_f32,xSpacing,"f");


static PyMethodDef arm_linear_interp_instance_f32_methods[] = {

    {"nValues", (PyCFunction) Method_arm_linear_interp_instance_f32_nValues,METH_NOARGS,"nValues"},
    {"x1", (PyCFunction) Method_arm_linear_interp_instance_f32_x1,METH_NOARGS,"x1"},
    {"xSpacing", (PyCFunction) Method_arm_linear_interp_instance_f32_xSpacing,METH_NOARGS,"xSpacing"},

    {NULL}  /* Sentinel */
};


DSPType(arm_linear_interp_instance_f32,arm_linear_interp_instance_f32_new,arm_linear_interp_instance_f32_dealloc,arm_linear_interp_instance_f32_init,arm_linear_interp_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_bilinear_interp_instance_f32 *instance;
} dsp_arm_bilinear_interp_instance_f32Object;


static void
arm_bilinear_interp_instance_f32_dealloc(dsp_arm_bilinear_interp_instance_f32Object* self)
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
arm_bilinear_interp_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_bilinear_interp_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_bilinear_interp_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_bilinear_interp_instance_f32));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_bilinear_interp_instance_f32_init(dsp_arm_bilinear_interp_instance_f32Object *self, PyObject *args, PyObject *kwds)
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

GETFIELD(arm_bilinear_interp_instance_f32,numRows,"h");
GETFIELD(arm_bilinear_interp_instance_f32,numCols,"h");


static PyMethodDef arm_bilinear_interp_instance_f32_methods[] = {

    {"numRows", (PyCFunction) Method_arm_bilinear_interp_instance_f32_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_bilinear_interp_instance_f32_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_bilinear_interp_instance_f32,arm_bilinear_interp_instance_f32_new,arm_bilinear_interp_instance_f32_dealloc,arm_bilinear_interp_instance_f32_init,arm_bilinear_interp_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_bilinear_interp_instance_q31 *instance;
} dsp_arm_bilinear_interp_instance_q31Object;


static void
arm_bilinear_interp_instance_q31_dealloc(dsp_arm_bilinear_interp_instance_q31Object* self)
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
arm_bilinear_interp_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_bilinear_interp_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_bilinear_interp_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_bilinear_interp_instance_q31));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_bilinear_interp_instance_q31_init(dsp_arm_bilinear_interp_instance_q31Object *self, PyObject *args, PyObject *kwds)
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

GETFIELD(arm_bilinear_interp_instance_q31,numRows,"h");
GETFIELD(arm_bilinear_interp_instance_q31,numCols,"h");


static PyMethodDef arm_bilinear_interp_instance_q31_methods[] = {

    {"numRows", (PyCFunction) Method_arm_bilinear_interp_instance_q31_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_bilinear_interp_instance_q31_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_bilinear_interp_instance_q31,arm_bilinear_interp_instance_q31_new,arm_bilinear_interp_instance_q31_dealloc,arm_bilinear_interp_instance_q31_init,arm_bilinear_interp_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_bilinear_interp_instance_q15 *instance;
} dsp_arm_bilinear_interp_instance_q15Object;


static void
arm_bilinear_interp_instance_q15_dealloc(dsp_arm_bilinear_interp_instance_q15Object* self)
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
arm_bilinear_interp_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_bilinear_interp_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_bilinear_interp_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_bilinear_interp_instance_q15));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_bilinear_interp_instance_q15_init(dsp_arm_bilinear_interp_instance_q15Object *self, PyObject *args, PyObject *kwds)
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

GETFIELD(arm_bilinear_interp_instance_q15,numRows,"h");
GETFIELD(arm_bilinear_interp_instance_q15,numCols,"h");


static PyMethodDef arm_bilinear_interp_instance_q15_methods[] = {

    {"numRows", (PyCFunction) Method_arm_bilinear_interp_instance_q15_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_bilinear_interp_instance_q15_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_bilinear_interp_instance_q15,arm_bilinear_interp_instance_q15_new,arm_bilinear_interp_instance_q15_dealloc,arm_bilinear_interp_instance_q15_init,arm_bilinear_interp_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_bilinear_interp_instance_q7 *instance;
} dsp_arm_bilinear_interp_instance_q7Object;


static void
arm_bilinear_interp_instance_q7_dealloc(dsp_arm_bilinear_interp_instance_q7Object* self)
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
arm_bilinear_interp_instance_q7_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_bilinear_interp_instance_q7Object *self;
    //printf("New called\n");

    self = (dsp_arm_bilinear_interp_instance_q7Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_bilinear_interp_instance_q7));

        self->instance->pData = NULL;

    }


    return (PyObject *)self;
}

static int
arm_bilinear_interp_instance_q7_init(dsp_arm_bilinear_interp_instance_q7Object *self, PyObject *args, PyObject *kwds)
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

    INITARRAYFIELD(pData,NPY_BYTE,int8_t,q7_t);

    }
    return 0;
}

GETFIELD(arm_bilinear_interp_instance_q7,numRows,"h");
GETFIELD(arm_bilinear_interp_instance_q7,numCols,"h");


static PyMethodDef arm_bilinear_interp_instance_q7_methods[] = {

    {"numRows", (PyCFunction) Method_arm_bilinear_interp_instance_q7_numRows,METH_NOARGS,"numRows"},
    {"numCols", (PyCFunction) Method_arm_bilinear_interp_instance_q7_numCols,METH_NOARGS,"numCols"},

    {NULL}  /* Sentinel */
};


DSPType(arm_bilinear_interp_instance_q7,arm_bilinear_interp_instance_q7_new,arm_bilinear_interp_instance_q7_dealloc,arm_bilinear_interp_instance_q7_init,arm_bilinear_interp_instance_q7_methods);


typedef struct {
    PyObject_HEAD
    arm_spline_instance_f32 *instance;
} dsp_arm_spline_instance_f32Object;


static void
arm_spline_instance_f32_dealloc(dsp_arm_spline_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->x)
       {
          PyMem_Free((float32_t *)self->instance->x);
       }

       if (self->instance->y)
       {
          PyMem_Free((float32_t *)self->instance->y);
       }

       if (self->instance->coeffs)
       {
          PyMem_Free((float32_t *)self->instance->coeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_spline_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_spline_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_spline_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_spline_instance_f32));

        self->instance->x = NULL;
        self->instance->y = NULL;
        self->instance->coeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_spline_instance_f32_init(dsp_arm_spline_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *x=NULL;
    PyObject *y=NULL;

char *kwlist[] = {
"type","x","y","n_x",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|iOOi", kwlist,&self->instance->type
,&x
,&y
,&self->instance->type
))
    {

    INITARRAYFIELD(x,NPY_DOUBLE,double,float32_t);
    INITARRAYFIELD(y,NPY_DOUBLE,double,float32_t);

    }
    return 0;
}

GETFIELD(arm_spline_instance_f32,type,"i");
GETFIELD(arm_spline_instance_f32,n_x,"i");


static PyMethodDef arm_spline_instance_f32_methods[] = {

    {"type", (PyCFunction) Method_arm_spline_instance_f32_type,METH_NOARGS,"type"},
    {"n_x", (PyCFunction) Method_arm_spline_instance_f32_n_x,METH_NOARGS,"n_x"},

    {NULL}  /* Sentinel */
};


DSPType(arm_spline_instance_f32,arm_spline_instance_f32_new,arm_spline_instance_f32_dealloc,arm_spline_instance_f32_init,arm_spline_instance_f32_methods);



void typeRegistration(PyObject *module) {

  
  
 
  ADDTYPE(arm_linear_interp_instance_f32);
  ADDTYPE(arm_bilinear_interp_instance_f32);
  ADDTYPE(arm_bilinear_interp_instance_q31);
  ADDTYPE(arm_bilinear_interp_instance_q15);
  ADDTYPE(arm_bilinear_interp_instance_q7);
  ADDTYPE(arm_spline_instance_f32);

}





static PyObject *
cmsis_arm_linear_interp_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  float32_t x; // input

  if (PyArg_ParseTuple(args,"Of",&S,&x))
  {

    dsp_arm_linear_interp_instance_f32Object *selfS = (dsp_arm_linear_interp_instance_f32Object *)S;

    float32_t returnValue = arm_linear_interp_f32(selfS->instance,x);
    PyObject* theReturnOBJ=Py_BuildValue("f",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_linear_interp_q31(PyObject *obj, PyObject *args)
{

  PyObject *pYData=NULL; // input
  q31_t *pYData_converted=NULL; // input
  q31_t x; // input
  uint32_t nValues; // input

  if (PyArg_ParseTuple(args,"Oii",&pYData,&x,&nValues))
  {

    GETARGUMENT(pYData,NPY_INT32,int32_t,int32_t);

    q31_t returnValue = arm_linear_interp_q31(pYData_converted,x,nValues);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pYData_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_linear_interp_q15(PyObject *obj, PyObject *args)
{

  PyObject *pYData=NULL; // input
  q15_t *pYData_converted=NULL; // input
  q31_t x; // input
  uint32_t nValues; // input

  if (PyArg_ParseTuple(args,"Oii",&pYData,&x,&nValues))
  {

    GETARGUMENT(pYData,NPY_INT16,int16_t,int16_t);

    q15_t returnValue = arm_linear_interp_q15(pYData_converted,x,nValues);
    PyObject* theReturnOBJ=Py_BuildValue("h",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pYData_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_linear_interp_q7(PyObject *obj, PyObject *args)
{

  PyObject *pYData=NULL; // input
  q7_t *pYData_converted=NULL; // input
  q31_t x; // input
  uint32_t nValues; // input

  if (PyArg_ParseTuple(args,"Oii",&pYData,&x,&nValues))
  {

    GETARGUMENT(pYData,NPY_BYTE,int8_t,q7_t);

    q7_t returnValue = arm_linear_interp_q7(pYData_converted,x,nValues);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pYData_converted);
    return(pythonResult);

  }
  return(NULL);
}




static PyObject *
cmsis_arm_bilinear_interp_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  float32_t X; // input
  float32_t Y; // input

  if (PyArg_ParseTuple(args,"Off",&S,&X,&Y))
  {

    dsp_arm_bilinear_interp_instance_f32Object *selfS = (dsp_arm_bilinear_interp_instance_f32Object *)S;

    float32_t returnValue = arm_bilinear_interp_f32(selfS->instance,X,Y);
    PyObject* theReturnOBJ=Py_BuildValue("f",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_bilinear_interp_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  q31_t X; // input
  q31_t Y; // input

  if (PyArg_ParseTuple(args,"Oii",&S,&X,&Y))
  {

    dsp_arm_bilinear_interp_instance_q31Object *selfS = (dsp_arm_bilinear_interp_instance_q31Object *)S;

    q31_t returnValue = arm_bilinear_interp_q31(selfS->instance,X,Y);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_bilinear_interp_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  q31_t X; // input
  q31_t Y; // input

  if (PyArg_ParseTuple(args,"Oii",&S,&X,&Y))
  {

    dsp_arm_bilinear_interp_instance_q15Object *selfS = (dsp_arm_bilinear_interp_instance_q15Object *)S;

    q15_t returnValue = arm_bilinear_interp_q15(selfS->instance,X,Y);
    PyObject* theReturnOBJ=Py_BuildValue("h",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_bilinear_interp_q7(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  q31_t X; // input
  q31_t Y; // input

  if (PyArg_ParseTuple(args,"Oii",&S,&X,&Y))
  {

    dsp_arm_bilinear_interp_instance_q7Object *selfS = (dsp_arm_bilinear_interp_instance_q7Object *)S;

    q7_t returnValue = arm_bilinear_interp_q7(selfS->instance,X,Y);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_arm_spline_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t type;
  PyObject *pX=NULL; // input
  float32_t *pX_converted=NULL; // input
  PyObject *pY=NULL; // input
  float32_t *pY_converted=NULL; // input
  uint32_t n;

  if (PyArg_ParseTuple(args,"OiOO",&S,&type,&pX,&pY))
  {

    dsp_arm_spline_instance_f32Object *selfS = (dsp_arm_spline_instance_f32Object *)S;

    GETARGUMENT(pX,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pY,NPY_DOUBLE,double,float32_t);
    n = arraySizepX ;
    float32_t * coeffs=PyMem_Malloc(sizeof(float32_t)*n*3);
    float32_t * tempBuffer=PyMem_Malloc(sizeof(float32_t)*n*2);

    arm_spline_init_f32(selfS->instance,
        type,pX_converted,pY_converted,n,coeffs,tempBuffer);

    PyObject* theReturnOBJ=Py_BuildValue("i",0);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    PyMem_Free(tempBuffer);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_spline_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_spline_instance_f32Object *selfS = (dsp_arm_spline_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_spline_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyMethodDef CMSISDSPMethods[] = {



{"arm_linear_interp_f32",  cmsis_arm_linear_interp_f32, METH_VARARGS,""},
{"arm_linear_interp_q31",  cmsis_arm_linear_interp_q31, METH_VARARGS,""},
{"arm_linear_interp_q15",  cmsis_arm_linear_interp_q15, METH_VARARGS,""},
{"arm_linear_interp_q7",  cmsis_arm_linear_interp_q7, METH_VARARGS,""},

{"arm_bilinear_interp_f32",  cmsis_arm_bilinear_interp_f32, METH_VARARGS,""},
{"arm_bilinear_interp_q31",  cmsis_arm_bilinear_interp_q31, METH_VARARGS,""},
{"arm_bilinear_interp_q15",  cmsis_arm_bilinear_interp_q15, METH_VARARGS,""},
{"arm_bilinear_interp_q7",  cmsis_arm_bilinear_interp_q7, METH_VARARGS,""},
{"arm_spline_f32",  cmsis_arm_spline_f32, METH_VARARGS,""},
{"arm_spline_init_f32",  cmsis_arm_spline_init_f32, METH_VARARGS,""},
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