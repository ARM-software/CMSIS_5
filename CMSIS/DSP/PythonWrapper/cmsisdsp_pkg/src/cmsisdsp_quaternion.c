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

#define MODNAME "cmsisdsp_quaternion"
#define MODINITNAME cmsisdsp_quaternion

#include "cmsisdsp_module.h"


NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);


void typeRegistration(PyObject *module) {

 
}







static PyObject *
cmsis_arm_quaternion_product_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbQuaternions; // input

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    nbQuaternions = arraySizepSrcA / 4 ;

    pDst=PyMem_Malloc(4*sizeof(float32_t)*nbQuaternions);


    arm_quaternion_product_f32(pSrcA_converted,pSrcB_converted,pDst,nbQuaternions);
 FLOATARRAY1(pDstOBJ,4*nbQuaternions,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_quaternion_product_single_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  float32_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OO",&pSrcA,&pSrcB))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);

    pDst=PyMem_Malloc(4*sizeof(float32_t));


    arm_quaternion_product_single_f32(pSrcA_converted,pSrcB_converted,pDst);
 FLOATARRAY1(pDstOBJ,4,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_arm_quaternion2rotation_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbQuaternions; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    nbQuaternions = arraySizepSrc / 4 ;

    pDst=PyMem_Malloc(9*sizeof(float32_t)*nbQuaternions);


    arm_quaternion2rotation_f32(pSrc_converted,pDst,nbQuaternions);
 FLOATARRAY1(pDstOBJ,9*nbQuaternions,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_rotation2quaternion_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbQuaternions; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    nbQuaternions = arraySizepSrc / 9 ;

    pDst=PyMem_Malloc(4*sizeof(float32_t)*nbQuaternions);


    arm_rotation2quaternion_f32(pSrc_converted,pDst,nbQuaternions);
 FLOATARRAY1(pDstOBJ,4*nbQuaternions,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_quaternion_normalize_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbQuaternions; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    nbQuaternions = arraySizepSrc / 4 ;

    pDst=PyMem_Malloc(4*sizeof(float32_t)*nbQuaternions);


    arm_quaternion_normalize_f32(pSrc_converted,pDst,nbQuaternions);
 FLOATARRAY1(pDstOBJ,4*nbQuaternions,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_quaternion_norm_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbQuaternions; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    nbQuaternions = arraySizepSrc / 4 ;

    pDst=PyMem_Malloc(sizeof(float32_t)*nbQuaternions);


    arm_quaternion_norm_f32(pSrc_converted,pDst,nbQuaternions);
 FLOATARRAY1(pDstOBJ,nbQuaternions,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_quaternion_conjugate_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbQuaternions; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    nbQuaternions = arraySizepSrc / 4 ;

    pDst=PyMem_Malloc(4*sizeof(float32_t)*nbQuaternions);


    arm_quaternion_conjugate_f32(pSrc_converted,pDst,nbQuaternions);
 FLOATARRAY1(pDstOBJ,4*nbQuaternions,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_quaternion_inverse_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbQuaternions; // input

  if (PyArg_ParseTuple(args,"O",&pSrc))
  {

    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    nbQuaternions = arraySizepSrc / 4 ;

    pDst=PyMem_Malloc(4*sizeof(float32_t)*nbQuaternions);


    arm_quaternion_inverse_f32(pSrc_converted,pDst,nbQuaternions);
 FLOATARRAY1(pDstOBJ,4*nbQuaternions,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}






static PyMethodDef CMSISDSPMethods[] = {






{"arm_quaternion_normalize_f32" ,  cmsis_arm_quaternion_normalize_f32, METH_VARARGS,""},
{"arm_quaternion_conjugate_f32" ,  cmsis_arm_quaternion_conjugate_f32, METH_VARARGS,""},
{"arm_quaternion_inverse_f32" ,  cmsis_arm_quaternion_inverse_f32, METH_VARARGS,""},
{"arm_quaternion_norm_f32" ,  cmsis_arm_quaternion_norm_f32, METH_VARARGS,""},
{"arm_quaternion2rotation_f32" ,  cmsis_arm_quaternion2rotation_f32, METH_VARARGS,""},
{"arm_rotation2quaternion_f32" ,  cmsis_arm_rotation2quaternion_f32, METH_VARARGS,""},
{"arm_quaternion_product_f32" ,  cmsis_arm_quaternion_product_f32, METH_VARARGS,""},
{"arm_quaternion_product_single_f32" ,  cmsis_arm_quaternion_product_single_f32, METH_VARARGS,""},


   
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