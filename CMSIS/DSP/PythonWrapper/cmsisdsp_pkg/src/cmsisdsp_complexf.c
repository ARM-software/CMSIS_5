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

#define MODNAME "cmsisdsp_complexf"
#define MODINITNAME cmsisdsp_complexf

#include "cmsisdsp_module.h"


NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);


void typeRegistration(PyObject *module) {

 
}











static PyObject *
cmsis_arm_cmplx_conj_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(float32_t)*2*numSamples);


    arm_cmplx_conj_f32(pSrc_converted,pDst,numSamples);
 FLOATARRAY1(pDstOBJ,2*numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_conj_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q31_t)*2*numSamples);


    arm_cmplx_conj_q31(pSrc_converted,pDst,numSamples);
 INT32ARRAY1(pDstOBJ,2*numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_conj_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q15_t)*2*numSamples);


    arm_cmplx_conj_q15(pSrc_converted,pDst,numSamples);
 INT16ARRAY1(pDstOBJ,2*numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mag_squared_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(float32_t)*2*numSamples);


    arm_cmplx_mag_squared_f32(pSrc_converted,pDst,numSamples);
 FLOATARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_cmplx_mag_squared_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float64_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(float64_t)*2*numSamples);


    arm_cmplx_mag_squared_f64(pSrc_converted,pDst,numSamples);
 FLOAT64ARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mag_squared_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q31_t)*2*numSamples);


    arm_cmplx_mag_squared_q31(pSrc_converted,pDst,numSamples);
 INT32ARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mag_squared_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q15_t)*2*numSamples);


    arm_cmplx_mag_squared_q15(pSrc_converted,pDst,numSamples);
 INT16ARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}







static PyObject *
cmsis_arm_cmplx_mag_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(float32_t)*2*numSamples);


    arm_cmplx_mag_f32(pSrc_converted,pDst,numSamples);
 FLOATARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_cmplx_mag_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float64_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(float64_t)*2*numSamples);


    arm_cmplx_mag_f64(pSrc_converted,pDst,numSamples);
 FLOAT64ARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mag_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q31_t)*2*numSamples);


    arm_cmplx_mag_q31(pSrc_converted,pDst,numSamples);
 INT32ARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mag_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q15_t)*2*numSamples);


    arm_cmplx_mag_q15(pSrc_converted,pDst,numSamples);
 INT16ARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_cmplx_mag_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    numSamples = arraySizepSrc ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q15_t)*2*numSamples);


    arm_cmplx_mag_fast_q15(pSrc_converted,pDst,numSamples);
 INT16ARRAY1(pDstOBJ,numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_cmplx_dot_prod_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t numSamples; // input
  q31_t realResult; // output
  q31_t imagResult; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    numSamples = arraySizepSrcA ;
    numSamples = numSamples / 2;


    arm_cmplx_dot_prod_q15(pSrcA_converted,pSrcB_converted,numSamples,&realResult,&imagResult);
    PyObject* realResultOBJ=Py_BuildValue("i",realResult);
    PyObject* imagResultOBJ=Py_BuildValue("i",imagResult);

    PyObject *pythonResult = Py_BuildValue("OO",realResultOBJ,imagResultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(realResultOBJ);
    Py_DECREF(imagResultOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_dot_prod_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t numSamples; // input
  q63_t realResult; // output
  q63_t imagResult; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    numSamples = arraySizepSrcA ;
    numSamples = numSamples / 2;


    arm_cmplx_dot_prod_q31(pSrcA_converted,pSrcB_converted,numSamples,&realResult,&imagResult);
    PyObject* realResultOBJ=Py_BuildValue("L",realResult);
    PyObject* imagResultOBJ=Py_BuildValue("L",imagResult);

    PyObject *pythonResult = Py_BuildValue("OO",realResultOBJ,imagResultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(realResultOBJ);
    Py_DECREF(imagResultOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_dot_prod_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  uint32_t numSamples; // input
  float32_t realResult; // output
  float32_t imagResult; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    numSamples = arraySizepSrcA ;
    numSamples = numSamples / 2;


    arm_cmplx_dot_prod_f32(pSrcA_converted,pSrcB_converted,numSamples,&realResult,&imagResult);
    PyObject* realResultOBJ=Py_BuildValue("f",realResult);
    PyObject* imagResultOBJ=Py_BuildValue("f",imagResult);

    PyObject *pythonResult = Py_BuildValue("OO",realResultOBJ,imagResultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(realResultOBJ);
    Py_DECREF(imagResultOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mult_real_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcCmplx=NULL; // input
  q15_t *pSrcCmplx_converted=NULL; // input
  PyObject *pSrcReal=NULL; // input
  q15_t *pSrcReal_converted=NULL; // input
  q15_t *pCmplxDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcCmplx,&pSrcReal))
  {

    GETARGUMENT(pSrcCmplx,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcReal,NPY_INT16,int16_t,int16_t);
    numSamples = arraySizepSrcCmplx ;
    numSamples = numSamples / 2;

    pCmplxDst=PyMem_Malloc(sizeof(q15_t)*2*numSamples);


    arm_cmplx_mult_real_q15(pSrcCmplx_converted,pSrcReal_converted,pCmplxDst,numSamples);
 INT16ARRAY1(pCmplxDstOBJ,2*numSamples,pCmplxDst);

    PyObject *pythonResult = Py_BuildValue("O",pCmplxDstOBJ);

    FREEARGUMENT(pSrcCmplx_converted);
    FREEARGUMENT(pSrcReal_converted);
    Py_DECREF(pCmplxDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mult_real_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcCmplx=NULL; // input
  q31_t *pSrcCmplx_converted=NULL; // input
  PyObject *pSrcReal=NULL; // input
  q31_t *pSrcReal_converted=NULL; // input
  q31_t *pCmplxDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcCmplx,&pSrcReal))
  {

    GETARGUMENT(pSrcCmplx,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcReal,NPY_INT32,int32_t,int32_t);
    numSamples = arraySizepSrcCmplx ;
    numSamples = numSamples / 2;

    pCmplxDst=PyMem_Malloc(sizeof(q31_t)*2*numSamples);


    arm_cmplx_mult_real_q31(pSrcCmplx_converted,pSrcReal_converted,pCmplxDst,numSamples);
 INT32ARRAY1(pCmplxDstOBJ,2*numSamples,pCmplxDst);

    PyObject *pythonResult = Py_BuildValue("O",pCmplxDstOBJ);

    FREEARGUMENT(pSrcCmplx_converted);
    FREEARGUMENT(pSrcReal_converted);
    Py_DECREF(pCmplxDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mult_real_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcCmplx=NULL; // input
  float32_t *pSrcCmplx_converted=NULL; // input
  PyObject *pSrcReal=NULL; // input
  float32_t *pSrcReal_converted=NULL; // input
  float32_t *pCmplxDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcCmplx,&pSrcReal))
  {

    GETARGUMENT(pSrcCmplx,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcReal,NPY_DOUBLE,double,float32_t);
    numSamples = arraySizepSrcCmplx ;
    numSamples = numSamples / 2;

    pCmplxDst=PyMem_Malloc(sizeof(float32_t)*2*numSamples);


    arm_cmplx_mult_real_f32(pSrcCmplx_converted,pSrcReal_converted,pCmplxDst,numSamples);
 FLOATARRAY1(pCmplxDstOBJ,2*numSamples,pCmplxDst);

    PyObject *pythonResult = Py_BuildValue("O",pCmplxDstOBJ);

    FREEARGUMENT(pSrcCmplx_converted);
    FREEARGUMENT(pSrcReal_converted);
    Py_DECREF(pCmplxDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}




static PyObject *
cmsis_arm_cmplx_mult_cmplx_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    numSamples = arraySizepSrcA ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q15_t)*2*numSamples);


    arm_cmplx_mult_cmplx_q15(pSrcA_converted,pSrcB_converted,pDst,numSamples);
 INT16ARRAY1(pDstOBJ,2*numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mult_cmplx_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    numSamples = arraySizepSrcA ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(q31_t)*2*numSamples);


    arm_cmplx_mult_cmplx_q31(pSrcA_converted,pSrcB_converted,pDst,numSamples);
 INT32ARRAY1(pDstOBJ,2*numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_cmplx_mult_cmplx_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    numSamples = arraySizepSrcA ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(float32_t)*2*numSamples);


    arm_cmplx_mult_cmplx_f32(pSrcA_converted,pSrcB_converted,pDst,numSamples);
 FLOATARRAY1(pDstOBJ,2*numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_cmplx_mult_cmplx_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float64_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float64_t *pSrcB_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t numSamples; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float64_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float64_t);
    numSamples = arraySizepSrcA ;
    numSamples = numSamples / 2;

    pDst=PyMem_Malloc(sizeof(float64_t)*2*numSamples);


    arm_cmplx_mult_cmplx_f64(pSrcA_converted,pSrcB_converted,pDst,numSamples);
 FLOAT64ARRAY1(pDstOBJ,2*numSamples,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyMethodDef CMSISDSPMethods[] = {


{"arm_cmplx_conj_f32",  cmsis_arm_cmplx_conj_f32, METH_VARARGS,""},
{"arm_cmplx_conj_q31",  cmsis_arm_cmplx_conj_q31, METH_VARARGS,""},
{"arm_cmplx_conj_q15",  cmsis_arm_cmplx_conj_q15, METH_VARARGS,""},
{"arm_cmplx_mag_squared_f32",  cmsis_arm_cmplx_mag_squared_f32, METH_VARARGS,""},
{"arm_cmplx_mag_squared_f64",  cmsis_arm_cmplx_mag_squared_f64, METH_VARARGS,""},
{"arm_cmplx_mag_squared_q31",  cmsis_arm_cmplx_mag_squared_q31, METH_VARARGS,""},
{"arm_cmplx_mag_squared_q15",  cmsis_arm_cmplx_mag_squared_q15, METH_VARARGS,""},


{"arm_cmplx_mag_f32",  cmsis_arm_cmplx_mag_f32, METH_VARARGS,""},
{"arm_cmplx_mag_f64",  cmsis_arm_cmplx_mag_f64, METH_VARARGS,""},
{"arm_cmplx_mag_q31",  cmsis_arm_cmplx_mag_q31, METH_VARARGS,""},
{"arm_cmplx_mag_q15",  cmsis_arm_cmplx_mag_q15, METH_VARARGS,""},
{"arm_cmplx_mag_fast_q15",  cmsis_arm_cmplx_mag_fast_q15, METH_VARARGS,""},
{"arm_cmplx_dot_prod_q15",  cmsis_arm_cmplx_dot_prod_q15, METH_VARARGS,""},
{"arm_cmplx_dot_prod_q31",  cmsis_arm_cmplx_dot_prod_q31, METH_VARARGS,""},
{"arm_cmplx_dot_prod_f32",  cmsis_arm_cmplx_dot_prod_f32, METH_VARARGS,""},
{"arm_cmplx_mult_real_q15",  cmsis_arm_cmplx_mult_real_q15, METH_VARARGS,""},
{"arm_cmplx_mult_real_q31",  cmsis_arm_cmplx_mult_real_q31, METH_VARARGS,""},
{"arm_cmplx_mult_real_f32",  cmsis_arm_cmplx_mult_real_f32, METH_VARARGS,""},
{"arm_cmplx_mult_cmplx_q15",  cmsis_arm_cmplx_mult_cmplx_q15, METH_VARARGS,""},
{"arm_cmplx_mult_cmplx_q31",  cmsis_arm_cmplx_mult_cmplx_q31, METH_VARARGS,""},
{"arm_cmplx_mult_cmplx_f32",  cmsis_arm_cmplx_mult_cmplx_f32, METH_VARARGS,""},
{"arm_cmplx_mult_cmplx_f64",  cmsis_arm_cmplx_mult_cmplx_f64, METH_VARARGS,""},

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