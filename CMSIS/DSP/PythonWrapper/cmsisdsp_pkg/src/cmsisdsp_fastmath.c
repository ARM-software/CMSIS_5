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

#define MODNAME "cmsisdsp_fastmath"
#define MODINITNAME cmsisdsp_fastmath

#include "cmsisdsp_module.h"


NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);




void typeRegistration(PyObject *module) {

 
}



static PyObject *
cmsis_arm_vlog_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_vlog_q15(pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}




static PyObject *
cmsis_arm_vlog_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_vlog_q31(pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_arm_sin_f32(PyObject *obj, PyObject *args)
{

  float32_t x; // input

  if (PyArg_ParseTuple(args,"f",&x))
  {


    float32_t returnValue = arm_sin_f32(x);
    PyObject* theReturnOBJ=Py_BuildValue("f",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sin_q31(PyObject *obj, PyObject *args)
{

  q31_t x; // input

  if (PyArg_ParseTuple(args,"i",&x))
  {


    q31_t returnValue = arm_sin_q31(x);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sin_q15(PyObject *obj, PyObject *args)
{

  q15_t x; // input

  if (PyArg_ParseTuple(args,"h",&x))
  {


    q15_t returnValue = arm_sin_q15(x);
    PyObject* theReturnOBJ=Py_BuildValue("h",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cos_f32(PyObject *obj, PyObject *args)
{

  float32_t x; // input

  if (PyArg_ParseTuple(args,"f",&x))
  {


    float32_t returnValue = arm_cos_f32(x);
    PyObject* theReturnOBJ=Py_BuildValue("f",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cos_q31(PyObject *obj, PyObject *args)
{

  q31_t x; // input

  if (PyArg_ParseTuple(args,"i",&x))
  {


    q31_t returnValue = arm_cos_q31(x);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cos_q15(PyObject *obj, PyObject *args)
{

  q15_t x; // input

  if (PyArg_ParseTuple(args,"h",&x))
  {


    q15_t returnValue = arm_cos_q15(x);
    PyObject* theReturnOBJ=Py_BuildValue("h",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sqrt_f32(PyObject *obj, PyObject *args)
{

  float32_t in; // input
  float32_t pOut; // output

  if (PyArg_ParseTuple(args,"f",&in))
  {


    arm_status returnValue = arm_sqrt_f32(in,&pOut);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("f",pOut);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pOutOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);

    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sqrt_q31(PyObject *obj, PyObject *args)
{

  q31_t in; // input
  q31_t pOut; // output

  if (PyArg_ParseTuple(args,"i",&in))
  {




    arm_status returnValue = arm_sqrt_q31(in,&pOut);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("i",pOut);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pOutOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_divide_q31(PyObject *obj, PyObject *args)
{

  q31_t num,den; // input
  q31_t pOut;
  int16_t shift; // output

  if (PyArg_ParseTuple(args,"ii",&num,&den))
  {



    arm_status returnValue = arm_divide_q31(num,den,&pOut,&shift);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("i",pOut);
    PyObject* pShiftOBJ=Py_BuildValue("h",shift);

    PyObject *pythonResult = Py_BuildValue("OOO",theReturnOBJ,pOutOBJ,pShiftOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);
    Py_DECREF(pShiftOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_divide_q15(PyObject *obj, PyObject *args)
{

  q15_t num,den; // input
  q15_t pOut;
  int16_t shift; // output

  if (PyArg_ParseTuple(args,"hh",&num,&den))
  {



    arm_status returnValue = arm_divide_q15(num,den,&pOut,&shift);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("h",pOut);
    PyObject* pShiftOBJ=Py_BuildValue("h",shift);

    PyObject *pythonResult = Py_BuildValue("OOO",theReturnOBJ,pOutOBJ,pShiftOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);
    Py_DECREF(pShiftOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_sqrt_q15(PyObject *obj, PyObject *args)
{

  q15_t in; // input
  q15_t pOut; // output

  if (PyArg_ParseTuple(args,"h",&in))
  {



    arm_status returnValue = arm_sqrt_q15(in,&pOut);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("h",pOut);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pOutOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_vexp_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_vexp_f32(pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_vexp_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_vexp_f64(pSrc_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_vlog_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_vlog_f32(pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_vlog_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_vlog_f64(pSrc_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_atan2_q31(PyObject *obj, PyObject *args)
{

  q31_t x,y; // input
  q31_t pOut;

  if (PyArg_ParseTuple(args,"ii",&y,&x))
  {



    arm_status returnValue = arm_atan2_q31(y,x,&pOut);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("i",pOut);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pOutOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_atan2_q15(PyObject *obj, PyObject *args)
{

  q15_t x,y; // input
  q15_t pOut;

  if (PyArg_ParseTuple(args,"hh",&y,&x))
  {



    arm_status returnValue = arm_atan2_q15(y,x,&pOut);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("h",pOut);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pOutOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_atan2_f32(PyObject *obj, PyObject *args)
{

  float32_t x,y; // input
  float32_t pOut;

  if (PyArg_ParseTuple(args,"ff",&y,&x))
  {



    arm_status returnValue = arm_atan2_f32(y,x,&pOut);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* pOutOBJ=Py_BuildValue("f",pOut);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pOutOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(pOutOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyMethodDef CMSISDSPMethods[] = {




{"arm_vlog_q15",  cmsis_arm_vlog_q15, METH_VARARGS,""},
{"arm_vlog_q31",  cmsis_arm_vlog_q31, METH_VARARGS,""},


{"arm_sin_f32",  cmsis_arm_sin_f32, METH_VARARGS,""},
{"arm_sin_q31",  cmsis_arm_sin_q31, METH_VARARGS,""},
{"arm_sin_q15",  cmsis_arm_sin_q15, METH_VARARGS,""},
{"arm_cos_f32",  cmsis_arm_cos_f32, METH_VARARGS,""},
{"arm_cos_q31",  cmsis_arm_cos_q31, METH_VARARGS,""},
{"arm_cos_q15",  cmsis_arm_cos_q15, METH_VARARGS,""},
{"arm_sqrt_f32",  cmsis_arm_sqrt_f32, METH_VARARGS,""},
{"arm_sqrt_q31",  cmsis_arm_sqrt_q31, METH_VARARGS,""},
{"arm_sqrt_q15",  cmsis_arm_sqrt_q15, METH_VARARGS,""},
{"arm_divide_q31",  cmsis_arm_divide_q31, METH_VARARGS,""},
{"arm_divide_q15",  cmsis_arm_divide_q15, METH_VARARGS,""},
{"arm_vexp_f32",  cmsis_arm_vexp_f32, METH_VARARGS,""},
{"arm_vlog_f32",  cmsis_arm_vlog_f32, METH_VARARGS,""},
{"arm_vexp_f64",  cmsis_arm_vexp_f64, METH_VARARGS,""},
{"arm_vlog_f64",  cmsis_arm_vlog_f64, METH_VARARGS,""},
{"arm_atan2_f32",  cmsis_arm_atan2_f32, METH_VARARGS,""},
{"arm_atan2_q31",  cmsis_arm_atan2_q31, METH_VARARGS,""},
{"arm_atan2_q15",  cmsis_arm_atan2_q15, METH_VARARGS,""},
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