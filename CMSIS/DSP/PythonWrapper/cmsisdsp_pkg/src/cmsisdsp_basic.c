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

#define MODNAME "cmsisdsp_basic"
#define MODINITNAME cmsisdsp_basic

#include "cmsisdsp_module.h"


NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);


void typeRegistration(PyObject *module) {

 
}


static PyObject *
cmsis_arm_recip_q31(PyObject *obj, PyObject *args)
{

  q31_t in; // input
  q31_t dst; // output
  PyObject *pRecipTable=NULL; // input
  q31_t *pRecipTable_converted=NULL; // input

  if (PyArg_ParseTuple(args,"iO",&in,&pRecipTable))
  {

    GETARGUMENT(pRecipTable,NPY_INT32,int32_t,int32_t);


    uint32_t returnValue = arm_recip_q31(in,&dst,pRecipTable_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* dstOBJ=Py_BuildValue("i",dst);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,dstOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(dstOBJ);
    FREEARGUMENT(pRecipTable_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_recip_q15(PyObject *obj, PyObject *args)
{

  q15_t in; // input
  q15_t dst; // output
  PyObject *pRecipTable=NULL; // input
  q15_t *pRecipTable_converted=NULL; // input

  if (PyArg_ParseTuple(args,"hO",&in,&pRecipTable))
  {

    GETARGUMENT(pRecipTable,NPY_INT16,int16_t,int16_t);


    uint32_t returnValue = arm_recip_q15(in,&dst,pRecipTable_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
    PyObject* dstOBJ=Py_BuildValue("h",dst);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,dstOBJ);

    Py_DECREF(theReturnOBJ);
    Py_DECREF(dstOBJ);
    FREEARGUMENT(pRecipTable_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mult_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_mult_q7(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mult_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_mult_q15(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mult_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_mult_q31(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_mult_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_mult_f32(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_mult_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float64_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float64_t *pSrcB_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float64_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_mult_f64(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_arm_add_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_add_f32(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_add_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float64_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float64_t *pSrcB_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float64_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_add_f64(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

/*

For the arm_(and|xor|or)_u(32|16|8)

*/

#define U_UN_OP(OP,TYP,EXT,NPYTYPE)                     \
static PyObject *                                       \
cmsis_arm_##OP##_##EXT(PyObject *obj, PyObject *args)   \
{                                                       \
                                                        \
  PyObject *pSrcA=NULL;                                 \
  TYP *pSrcA_converted=NULL;                            \
  TYP *pDst=NULL;                                       \
  uint32_t blockSize;                                   \
                                                        \
  if (PyArg_ParseTuple(args,"O",&pSrcA))                \
  {                                                     \
                                                        \
    GETARGUMENT(pSrcA,NPYTYPE,TYP,TYP);                 \
    blockSize = arraySizepSrcA ;                        \
                                                        \
    pDst=PyMem_Malloc(sizeof(TYP)*blockSize);           \
                                                        \
                                                        \
    arm_##OP##_##EXT(pSrcA_converted,pDst,blockSize);   \
 TYP_ARRAY1(pDstOBJ,blockSize,pDst,NPYTYPE);            \
                                                        \
    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);\
                                                        \
    FREEARGUMENT(pSrcA_converted);                      \
    Py_DECREF(pDstOBJ);                                 \
    return(pythonResult);                               \
                                                        \
  }                                                     \
  return(NULL);                                         \
}

#define U_BIN_OP(OP,TYP,EXT,NPYTYPE)                                 \
static PyObject *                                                    \
cmsis_arm_##OP##_##EXT(PyObject *obj, PyObject *args)                \
{                                                                    \
                                                                     \
  PyObject *pSrcA=NULL;                                              \
  TYP *pSrcA_converted=NULL;                                         \
  PyObject *pSrcB=NULL;                                              \
  TYP *pSrcB_converted=NULL;                                         \
  TYP *pDst=NULL;                                                    \
  uint32_t blockSize;                                                \
                                                                     \
  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))                     \
  {                                                                  \
                                                                     \
    GETARGUMENT(pSrcA,NPYTYPE,TYP,TYP);                              \
    GETARGUMENT(pSrcB,NPYTYPE,TYP,TYP);                              \
    blockSize = arraySizepSrcA ;                                     \
                                                                     \
    pDst=PyMem_Malloc(sizeof(TYP)*blockSize);                        \
                                                                     \
                                                                     \
    arm_##OP##_##EXT(pSrcA_converted,pSrcB_converted,pDst,blockSize);\
 TYP_ARRAY1(pDstOBJ,blockSize,pDst,NPYTYPE);                         \
                                                                     \
    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);             \
                                                                     \
    FREEARGUMENT(pSrcA_converted);                                   \
    FREEARGUMENT(pSrcB_converted);                                   \
    Py_DECREF(pDstOBJ);                                              \
    return(pythonResult);                                            \
                                                                     \
  }                                                                  \
  return(NULL);                                                      \
}

U_BIN_OP(and,uint32_t,u32,NPY_UINT32);
U_BIN_OP(and,uint16_t,u16,NPY_UINT16);
U_BIN_OP(and,uint8_t,u8,NPY_UINT8);

U_BIN_OP(or,uint32_t,u32,NPY_UINT32);
U_BIN_OP(or,uint16_t,u16,NPY_UINT16);
U_BIN_OP(or,uint8_t,u8,NPY_UINT8);

U_BIN_OP(xor,uint32_t,u32,NPY_UINT32);
U_BIN_OP(xor,uint16_t,u16,NPY_UINT16);
U_BIN_OP(xor,uint8_t,u8,NPY_UINT8);

U_UN_OP(not,uint32_t,u32,NPY_UINT32);
U_UN_OP(not,uint16_t,u16,NPY_UINT16);
U_UN_OP(not,uint8_t,u8,NPY_UINT8);

static PyObject *
cmsis_arm_add_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_add_q7(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_add_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_add_q15(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_add_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_add_q31(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sub_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_sub_f32(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_sub_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float64_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float64_t *pSrcB_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float64_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_sub_f64(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sub_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_sub_q7(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sub_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_sub_q15(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sub_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrcA ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_sub_q31(pSrcA_converted,pSrcB_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_scale_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t scale; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Of",&pSrc,&scale))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_scale_f32(pSrc_converted,scale,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_scale_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t scale; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Od",&pSrc,&scale))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_scale_f64(pSrc_converted,scale,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_scale_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q7_t *pSrc_converted=NULL; // input
  int32_t scaleFract; // input
  int32_t shift; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oii",&pSrc,&scaleFract,&shift))
  {

    GETARGUMENT(pSrc,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_scale_q7(pSrc_converted,(q7_t)scaleFract,(int8_t)shift,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_scale_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t scaleFract; // input
  int32_t shift; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Ohi",&pSrc,&scaleFract,&shift))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_scale_q15(pSrc_converted,scaleFract,(int8_t)shift,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_scale_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t scaleFract; // input
  int32_t shift; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oii",&pSrc,&scaleFract,&shift))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_scale_q31(pSrc_converted,scaleFract,(int8_t)shift,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_abs_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q7_t *pSrc_converted=NULL; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_abs_q7(pSrc_converted,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_abs_f32(PyObject *obj, PyObject *args)
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


    arm_abs_f32(pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_abs_f64(PyObject *obj, PyObject *args)
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


    arm_abs_f64(pSrc_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_abs_q15(PyObject *obj, PyObject *args)
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


    arm_abs_q15(pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_abs_q31(PyObject *obj, PyObject *args)
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


    arm_abs_q31(pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}





static PyObject *
cmsis_arm_dot_prod_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  uint32_t blockSize; // input
  float32_t result; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrcA ;



    arm_dot_prod_f32(pSrcA_converted,pSrcB_converted,blockSize,&result);
    PyObject* resultOBJ=Py_BuildValue("f",result);

    PyObject *pythonResult = Py_BuildValue("O",resultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(resultOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_dot_prod_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float64_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float64_t *pSrcB_converted=NULL; // input
  uint32_t blockSize; // input
  float64_t result; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float64_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrcA ;



    arm_dot_prod_f64(pSrcA_converted,pSrcB_converted,blockSize,&result);
    PyObject* resultOBJ=Py_BuildValue("d",result);

    PyObject *pythonResult = Py_BuildValue("O",resultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(resultOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_dot_prod_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  uint32_t blockSize; // input
  q31_t result; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrcA ;



    arm_dot_prod_q7(pSrcA_converted,pSrcB_converted,blockSize,&result);
    PyObject* resultOBJ=Py_BuildValue("i",result);

    PyObject *pythonResult = Py_BuildValue("O",resultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(resultOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_dot_prod_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t blockSize; // input
  q63_t result; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrcA ;



    arm_dot_prod_q15(pSrcA_converted,pSrcB_converted,blockSize,&result);
    PyObject* resultOBJ=Py_BuildValue("L",result);

    PyObject *pythonResult = Py_BuildValue("O",resultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(resultOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_dot_prod_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t blockSize; // input
  q63_t result; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrcA ;



    arm_dot_prod_q31(pSrcA_converted,pSrcB_converted,blockSize,&result);
    PyObject* resultOBJ=Py_BuildValue("L",result);

    PyObject *pythonResult = Py_BuildValue("O",resultOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(resultOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_shift_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q7_t *pSrc_converted=NULL; // input
  int32_t shiftBits; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oi",&pSrc,&shiftBits))
  {

    GETARGUMENT(pSrc,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_shift_q7(pSrc_converted,(int8_t)shiftBits,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_shift_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  int32_t shiftBits; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oi",&pSrc,&shiftBits))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_shift_q15(pSrc_converted,(int8_t)shiftBits,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_shift_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  int32_t shiftBits; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oi",&pSrc,&shiftBits))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_shift_q31(pSrc_converted,(int8_t)shiftBits,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_clip_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t low,high; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Off",&pSrc,&low,&high))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_clip_f32(pSrc_converted,pDst,low,high,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_clip_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t low,high; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oii",&pSrc,&low,&high))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_clip_q31(pSrc_converted,pDst,low,high,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_clip_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t low,high; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Ohh",&pSrc,&low,&high))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_clip_q15(pSrc_converted,pDst,low,high,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_clip_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q7_t *pSrc_converted=NULL; // input
  int32_t low,high; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oii",&pSrc,&low,&high))
  {

    GETARGUMENT(pSrc,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_clip_q7(pSrc_converted,pDst,(q7_t)low,(q7_t)high,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_offset_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t offset; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Of",&pSrc,&offset))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_offset_f32(pSrc_converted,offset,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_offset_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t offset; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Od",&pSrc,&offset))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_offset_f64(pSrc_converted,offset,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_offset_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q7_t *pSrc_converted=NULL; // input
  int32_t offset; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oi",&pSrc,&offset))
  {

    GETARGUMENT(pSrc,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_offset_q7(pSrc_converted,(q7_t)offset,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_offset_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t offset; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oh",&pSrc,&offset))
  {

    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_offset_q15(pSrc_converted,offset,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_offset_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t offset; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"Oi",&pSrc,&offset))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_offset_q31(pSrc_converted,offset,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_negate_f32(PyObject *obj, PyObject *args)
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


    arm_negate_f32(pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_negate_f64(PyObject *obj, PyObject *args)
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


    arm_negate_f64(pSrc_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_negate_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  q7_t *pSrc_converted=NULL; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_negate_q7(pSrc_converted,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_negate_q15(PyObject *obj, PyObject *args)
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


    arm_negate_q15(pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_negate_q31(PyObject *obj, PyObject *args)
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


    arm_negate_q31(pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_ssat(PyObject *obj, PyObject *args)
{

  int32_t val; // input
  uint32_t sat;

  if (PyArg_ParseTuple(args,"iI",&val,&sat))
  {

    int32_t result = __SSAT(val,sat);
    PyObject* theReturnOBJ=Py_BuildValue("i",result);

    return(theReturnOBJ);

  }
  return(NULL);
}

static PyObject *
cmsis_usat(PyObject *obj, PyObject *args)
{

  int32_t val; // input
  uint32_t sat;
  q15_t pOut,shift; // output

  if (PyArg_ParseTuple(args,"iI",&val,&sat))
  {

    uint32_t result = __USAT(val,sat);
    PyObject* theReturnOBJ=Py_BuildValue("I",result);

    return(theReturnOBJ);

  }
  return(NULL);
}

static PyObject *
cmsis_clz(PyObject *obj, PyObject *args)
{

  uint32_t val;

  if (PyArg_ParseTuple(args,"I",&val))
  {

    uint8_t result = __CLZ(val);
    PyObject* theReturnOBJ=Py_BuildValue("B",result);

    return(theReturnOBJ);

  }
  return(NULL);
}

static PyMethodDef CMSISDSPMethods[] = {

{"arm_recip_q31",  cmsis_arm_recip_q31, METH_VARARGS,""},
{"arm_recip_q15",  cmsis_arm_recip_q15, METH_VARARGS,""},




{"arm_mult_q7",  cmsis_arm_mult_q7, METH_VARARGS,""},
{"arm_mult_q15",  cmsis_arm_mult_q15, METH_VARARGS,""},
{"arm_mult_q31",  cmsis_arm_mult_q31, METH_VARARGS,""},
{"arm_mult_f32",  cmsis_arm_mult_f32, METH_VARARGS,""},
{"arm_mult_f64",  cmsis_arm_mult_f64, METH_VARARGS,""},


{"arm_add_f32",  cmsis_arm_add_f32, METH_VARARGS,""},
{"arm_add_f64",  cmsis_arm_add_f64, METH_VARARGS,""},
{"arm_add_q7",  cmsis_arm_add_q7, METH_VARARGS,""},
{"arm_add_q15",  cmsis_arm_add_q15, METH_VARARGS,""},
{"arm_add_q31",  cmsis_arm_add_q31, METH_VARARGS,""},
{"arm_sub_f32",  cmsis_arm_sub_f32, METH_VARARGS,""},
{"arm_sub_f64",  cmsis_arm_sub_f64, METH_VARARGS,""},

{"arm_sub_q7",  cmsis_arm_sub_q7, METH_VARARGS,""},
{"arm_sub_q15",  cmsis_arm_sub_q15, METH_VARARGS,""},
{"arm_sub_q31",  cmsis_arm_sub_q31, METH_VARARGS,""},
{"arm_scale_f32",  cmsis_arm_scale_f32, METH_VARARGS,""},
{"arm_scale_f64",  cmsis_arm_scale_f64, METH_VARARGS,""},
{"arm_scale_q7",  cmsis_arm_scale_q7, METH_VARARGS,""},
{"arm_scale_q15",  cmsis_arm_scale_q15, METH_VARARGS,""},
{"arm_scale_q31",  cmsis_arm_scale_q31, METH_VARARGS,""},
{"arm_abs_q7",  cmsis_arm_abs_q7, METH_VARARGS,""},
{"arm_abs_f32",  cmsis_arm_abs_f32, METH_VARARGS,""},
{"arm_abs_f64",  cmsis_arm_abs_f64, METH_VARARGS,""},
{"arm_abs_q15",  cmsis_arm_abs_q15, METH_VARARGS,""},
{"arm_abs_q31",  cmsis_arm_abs_q31, METH_VARARGS,""},
{"arm_dot_prod_f32",  cmsis_arm_dot_prod_f32, METH_VARARGS,""},
{"arm_dot_prod_f64",  cmsis_arm_dot_prod_f64, METH_VARARGS,""},

{"arm_dot_prod_q7",  cmsis_arm_dot_prod_q7, METH_VARARGS,""},
{"arm_dot_prod_q15",  cmsis_arm_dot_prod_q15, METH_VARARGS,""},
{"arm_dot_prod_q31",  cmsis_arm_dot_prod_q31, METH_VARARGS,""},
{"arm_shift_q7",  cmsis_arm_shift_q7, METH_VARARGS,""},
{"arm_shift_q15",  cmsis_arm_shift_q15, METH_VARARGS,""},
{"arm_shift_q31",  cmsis_arm_shift_q31, METH_VARARGS,""},
{"arm_clip_f32",  cmsis_arm_clip_f32, METH_VARARGS,""},
{"arm_clip_q31",  cmsis_arm_clip_q31, METH_VARARGS,""},
{"arm_clip_q15",  cmsis_arm_clip_q15, METH_VARARGS,""},
{"arm_clip_q7",  cmsis_arm_clip_q7, METH_VARARGS,""},
{"arm_offset_f32",  cmsis_arm_offset_f32, METH_VARARGS,""},
{"arm_offset_f64",  cmsis_arm_offset_f64, METH_VARARGS,""},

{"arm_offset_q7",  cmsis_arm_offset_q7, METH_VARARGS,""},
{"arm_offset_q15",  cmsis_arm_offset_q15, METH_VARARGS,""},
{"arm_offset_q31",  cmsis_arm_offset_q31, METH_VARARGS,""},
{"arm_negate_f32",  cmsis_arm_negate_f32, METH_VARARGS,""},
{"arm_negate_f64",  cmsis_arm_negate_f64, METH_VARARGS,""},
{"arm_negate_q7",  cmsis_arm_negate_q7, METH_VARARGS,""},
{"arm_negate_q15",  cmsis_arm_negate_q15, METH_VARARGS,""},
{"arm_negate_q31",  cmsis_arm_negate_q31, METH_VARARGS,""},






{"arm_and_u32",  cmsis_arm_and_u32, METH_VARARGS,""},
{"arm_and_u16",  cmsis_arm_and_u16, METH_VARARGS,""},
{"arm_and_u8" ,  cmsis_arm_and_u8, METH_VARARGS,""},

{"arm_or_u32",  cmsis_arm_or_u32, METH_VARARGS,""},
{"arm_or_u16",  cmsis_arm_or_u16, METH_VARARGS,""},
{"arm_or_u8" ,  cmsis_arm_or_u8, METH_VARARGS,""},

{"arm_xor_u32",  cmsis_arm_xor_u32, METH_VARARGS,""},
{"arm_xor_u16",  cmsis_arm_xor_u16, METH_VARARGS,""},
{"arm_xor_u8" ,  cmsis_arm_xor_u8, METH_VARARGS,""},

{"arm_not_u32",  cmsis_arm_not_u32, METH_VARARGS,""},
{"arm_not_u16",  cmsis_arm_not_u16, METH_VARARGS,""},
{"arm_not_u8" ,  cmsis_arm_not_u8, METH_VARARGS,""},

   
    {"ssat",  cmsis_ssat, METH_VARARGS,""},
    {"usat",  cmsis_usat, METH_VARARGS,""},
    {"clz",  cmsis_clz, METH_VARARGS,""},
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