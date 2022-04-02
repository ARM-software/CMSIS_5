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

#define MODNAME "cmsisdsp_svm"
#define MODINITNAME cmsisdsp_svm

#include "cmsisdsp_module.h"



NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);


#define SVMTYPE(NAME)                       \
typedef struct {                            \
    PyObject_HEAD                           \
    arm_svm_##NAME##_instance_f32 *instance;\
} dsp_arm_svm_##NAME##_instance_f32Object;

SVMTYPE(linear);
SVMTYPE(polynomial);
SVMTYPE(rbf);
SVMTYPE(sigmoid);

#define SVMTYPEDEALLOC(NAME)                                                        \
static void                                                                         \
arm_svm_##NAME##_instance_f32_dealloc(dsp_arm_svm_##NAME##_instance_f32Object* self)\
{                                                                                   \
    if (self->instance)                                                             \
    {                                                                               \
                                                                                    \
       if (self->instance->dualCoefficients)                                        \
       {                                                                            \
          PyMem_Free((float32_t*)self->instance->dualCoefficients);                 \
       }                                                                            \
                                                                                    \
       if (self->instance->supportVectors)                                          \
       {                                                                            \
          PyMem_Free((float32_t*)self->instance->supportVectors);                   \
       }                                                                            \
                                                                                    \
       if (self->instance->classes)                                                 \
       {                                                                            \
          PyMem_Free((float32_t*)self->instance->classes);                          \
       }                                                                            \
                                                                                    \
       PyMem_Free(self->instance);                                                  \
    }                                                                               \
                                                                                    \
    Py_TYPE(self)->tp_free((PyObject*)self);                                        \
}

SVMTYPEDEALLOC(linear);
SVMTYPEDEALLOC(polynomial);
SVMTYPEDEALLOC(rbf);
SVMTYPEDEALLOC(sigmoid);

#define SVMTYPENEW(NAME)                                                             \
static PyObject *                                                                    \
arm_svm_##NAME##_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)\
{                                                                                    \
    dsp_arm_svm_##NAME##_instance_f32Object *self;                                   \
                                                                                     \
    self = (dsp_arm_svm_##NAME##_instance_f32Object *)type->tp_alloc(type, 0);       \
                                                                                     \
    if (self != NULL) {                                                              \
                                                                                     \
        self->instance = PyMem_Malloc(sizeof(arm_svm_##NAME##_instance_f32));        \
        self->instance->dualCoefficients=NULL;                                       \
        self->instance->supportVectors=NULL;                                         \
        self->instance->classes=NULL;                                                \
                                                                                     \
    }                                                                                \
                                                                                     \
                                                                                     \
    return (PyObject *)self;                                                         \
}

SVMTYPENEW(linear);
SVMTYPENEW(polynomial);
SVMTYPENEW(rbf);
SVMTYPENEW(sigmoid);

/*

LINEAR INIT 

*/

static int
arm_svm_linear_instance_f32_init(dsp_arm_svm_linear_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

PyObject *dualCoefficients=NULL;
PyObject *supportVectors=NULL;
PyObject *classes=NULL;

char *kwlist[] = {
"nbOfSupportVectors",
"vectorDimension",
"intercept",
"dualCoefficients",
"supportVectors",
"classes",
NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|kkfOOO", kwlist,
 &self->instance->nbOfSupportVectors
,&self->instance->vectorDimension
,&self->instance->intercept
,&dualCoefficients
,&supportVectors
,&classes
))
    {

      INITARRAYFIELD(dualCoefficients,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(supportVectors,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(classes,NPY_INT32,int32_t,int32_t);
    }
    return 0;
}

GETFIELD(arm_svm_linear_instance_f32,nbOfSupportVectors,"k");
GETFIELD(arm_svm_linear_instance_f32,vectorDimension,"k");
GETFIELD(arm_svm_linear_instance_f32,intercept,"f");


static PyMethodDef arm_svm_linear_instance_f32_methods[] = {

    {"nbOfSupportVectors", (PyCFunction) Method_arm_svm_linear_instance_f32_nbOfSupportVectors,METH_NOARGS,"nbOfSupportVectors"},
    {"vectorDimension", (PyCFunction) Method_arm_svm_linear_instance_f32_vectorDimension,METH_NOARGS,"vectorDimension"},
    {"intercept", (PyCFunction) Method_arm_svm_linear_instance_f32_intercept,METH_NOARGS,"intercept"},
   
    {NULL}  /* Sentinel */
};

/* 

POLYNOMIAl INIT

*/

static int
arm_svm_polynomial_instance_f32_init(dsp_arm_svm_polynomial_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

PyObject *dualCoefficients=NULL;
PyObject *supportVectors=NULL;
PyObject *classes=NULL;

char *kwlist[] = {
"nbOfSupportVectors",
"vectorDimension",
"intercept",
"dualCoefficients",
"supportVectors",
"classes",
"degree",
"coef0",
"gamma",
NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|kkfOOOiff", kwlist,
 &self->instance->nbOfSupportVectors
,&self->instance->vectorDimension
,&self->instance->intercept
,&dualCoefficients
,&supportVectors
,&classes
,&self->instance->degree
,&self->instance->coef0
,&self->instance->gamma
))
    {

      INITARRAYFIELD(dualCoefficients,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(supportVectors,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(classes,NPY_INT32,int32_t,int32_t);
    }
    return 0;
}

GETFIELD(arm_svm_polynomial_instance_f32,nbOfSupportVectors,"k");
GETFIELD(arm_svm_polynomial_instance_f32,vectorDimension,"k");
GETFIELD(arm_svm_polynomial_instance_f32,intercept,"f");
GETFIELD(arm_svm_polynomial_instance_f32,degree,"i");
GETFIELD(arm_svm_polynomial_instance_f32,coef0,"f");
GETFIELD(arm_svm_polynomial_instance_f32,gamma,"f");


static PyMethodDef arm_svm_polynomial_instance_f32_methods[] = {

    {"nbOfSupportVectors", (PyCFunction) Method_arm_svm_polynomial_instance_f32_nbOfSupportVectors,METH_NOARGS,"nbOfSupportVectors"},
    {"vectorDimension", (PyCFunction) Method_arm_svm_polynomial_instance_f32_vectorDimension,METH_NOARGS,"vectorDimension"},
    {"intercept", (PyCFunction) Method_arm_svm_polynomial_instance_f32_intercept,METH_NOARGS,"intercept"},
    {"degree", (PyCFunction) Method_arm_svm_polynomial_instance_f32_degree,METH_NOARGS,"degree"},
    {"coef0", (PyCFunction) Method_arm_svm_polynomial_instance_f32_coef0,METH_NOARGS,"coef0"},
    {"gamma", (PyCFunction) Method_arm_svm_polynomial_instance_f32_gamma,METH_NOARGS,"gamma"},

    {NULL}  /* Sentinel */
};

/*

RBF INIT

*/


static int
arm_svm_rbf_instance_f32_init(dsp_arm_svm_rbf_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

PyObject *dualCoefficients=NULL;
PyObject *supportVectors=NULL;
PyObject *classes=NULL;

char *kwlist[] = {
"nbOfSupportVectors",
"vectorDimension",
"intercept",
"dualCoefficients",
"supportVectors",
"classes",
"gamma",
NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|kkfOOOf", kwlist,
 &self->instance->nbOfSupportVectors
,&self->instance->vectorDimension
,&self->instance->intercept
,&dualCoefficients
,&supportVectors
,&classes
,&self->instance->gamma
))
    {

      INITARRAYFIELD(dualCoefficients,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(supportVectors,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(classes,NPY_INT32,int32_t,int32_t);
    }
    return 0;
}

GETFIELD(arm_svm_rbf_instance_f32,nbOfSupportVectors,"k");
GETFIELD(arm_svm_rbf_instance_f32,vectorDimension,"k");
GETFIELD(arm_svm_rbf_instance_f32,intercept,"f");
GETFIELD(arm_svm_rbf_instance_f32,gamma,"f");


static PyMethodDef arm_svm_rbf_instance_f32_methods[] = {

    {"nbOfSupportVectors", (PyCFunction) Method_arm_svm_rbf_instance_f32_nbOfSupportVectors,METH_NOARGS,"nbOfSupportVectors"},
    {"vectorDimension", (PyCFunction) Method_arm_svm_rbf_instance_f32_vectorDimension,METH_NOARGS,"vectorDimension"},
    {"intercept", (PyCFunction) Method_arm_svm_rbf_instance_f32_intercept,METH_NOARGS,"intercept"},
    {"gamma", (PyCFunction) Method_arm_svm_rbf_instance_f32_gamma,METH_NOARGS,"gamma"},

    {NULL}  /* Sentinel */
};

/*

SIGMOID INIT

*/

static int
arm_svm_sigmoid_instance_f32_init(dsp_arm_svm_sigmoid_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

PyObject *dualCoefficients=NULL;
PyObject *supportVectors=NULL;
PyObject *classes=NULL;

char *kwlist[] = {
"nbOfSupportVectors",
"vectorDimension",
"intercept",
"dualCoefficients",
"supportVectors",
"classes",
"coef0",
"gamma",
NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|kkfOOOff", kwlist,
 &self->instance->nbOfSupportVectors
,&self->instance->vectorDimension
,&self->instance->intercept
,&dualCoefficients
,&supportVectors
,&classes
,&self->instance->coef0
,&self->instance->gamma
))
    {

      INITARRAYFIELD(dualCoefficients,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(supportVectors,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(classes,NPY_INT32,int32_t,int32_t);
    }
    return 0;
}

GETFIELD(arm_svm_sigmoid_instance_f32,nbOfSupportVectors,"k");
GETFIELD(arm_svm_sigmoid_instance_f32,vectorDimension,"k");
GETFIELD(arm_svm_sigmoid_instance_f32,intercept,"f");
GETFIELD(arm_svm_sigmoid_instance_f32,coef0,"f");
GETFIELD(arm_svm_sigmoid_instance_f32,gamma,"f");


static PyMethodDef arm_svm_sigmoid_instance_f32_methods[] = {

    {"nbOfSupportVectors", (PyCFunction) Method_arm_svm_sigmoid_instance_f32_nbOfSupportVectors,METH_NOARGS,"nbOfSupportVectors"},
    {"vectorDimension", (PyCFunction) Method_arm_svm_sigmoid_instance_f32_vectorDimension,METH_NOARGS,"vectorDimension"},
    {"intercept", (PyCFunction) Method_arm_svm_sigmoid_instance_f32_intercept,METH_NOARGS,"intercept"},
    {"coef0", (PyCFunction) Method_arm_svm_sigmoid_instance_f32_coef0,METH_NOARGS,"coef0"},
    {"gamma", (PyCFunction) Method_arm_svm_sigmoid_instance_f32_gamma,METH_NOARGS,"gamma"},

    {NULL}  /* Sentinel */
};

DSPType(arm_svm_linear_instance_f32,arm_svm_linear_instance_f32_new,arm_svm_linear_instance_f32_dealloc,arm_svm_linear_instance_f32_init,arm_svm_linear_instance_f32_methods);
DSPType(arm_svm_polynomial_instance_f32,arm_svm_polynomial_instance_f32_new,arm_svm_polynomial_instance_f32_dealloc,arm_svm_polynomial_instance_f32_init,arm_svm_polynomial_instance_f32_methods);
DSPType(arm_svm_rbf_instance_f32,arm_svm_rbf_instance_f32_new,arm_svm_rbf_instance_f32_dealloc,arm_svm_rbf_instance_f32_init,arm_svm_rbf_instance_f32_methods);
DSPType(arm_svm_sigmoid_instance_f32,arm_svm_sigmoid_instance_f32_new,arm_svm_sigmoid_instance_f32_dealloc,arm_svm_sigmoid_instance_f32_init,arm_svm_sigmoid_instance_f32_methods);




void typeRegistration(PyObject *module) {

  
  
  ADDTYPE(arm_svm_linear_instance_f32);
  ADDTYPE(arm_svm_polynomial_instance_f32);
  ADDTYPE(arm_svm_rbf_instance_f32);
  ADDTYPE(arm_svm_sigmoid_instance_f32);

  
}

static PyObject *
cmsis_arm_svm_linear_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pdualCoefficients=NULL; // input
  float32_t *pdualCoefficients_converted=NULL; // input
  PyObject *psupportVectors=NULL; // input
  float32_t *psupportVectors_converted=NULL; // input
  PyObject *pclasses=NULL; // input
  int32_t *pclasses_converted=NULL; // input
  uint32_t blockSize; // input

  uint32_t nbOfSupportVectors;
  uint32_t vectorDimension;
  float32_t intercept;

  if (PyArg_ParseTuple(args,"OkkfOOO",&S,
    &nbOfSupportVectors,
    &vectorDimension,
    &intercept,
    &pdualCoefficients,
    &psupportVectors,
    &pclasses
    ))
  {

    dsp_arm_svm_linear_instance_f32Object *selfS = (dsp_arm_svm_linear_instance_f32Object *)S;
    GETARGUMENT(pdualCoefficients,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(psupportVectors,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pclasses,NPY_INT32,int32_t,int32_t);

    arm_svm_linear_init_f32(selfS->instance,
      nbOfSupportVectors,
      vectorDimension,
      intercept,
      pdualCoefficients_converted,
      psupportVectors_converted,
      pclasses_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}

static PyObject *
cmsis_arm_svm_polynomial_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pdualCoefficients=NULL; // input
  float32_t *pdualCoefficients_converted=NULL; // input
  PyObject *psupportVectors=NULL; // input
  float32_t *psupportVectors_converted=NULL; // input
  PyObject *pclasses=NULL; // input
  int32_t *pclasses_converted=NULL; // input
  uint32_t blockSize; // input

  uint32_t nbOfSupportVectors;
  uint32_t vectorDimension;
  float32_t intercept;
  int32_t         degree;                
  float32_t       coef0;          
  float32_t       gamma;

  if (PyArg_ParseTuple(args,"OkkfOOOiff",&S,
    &nbOfSupportVectors,
    &vectorDimension,
    &intercept,
    &pdualCoefficients,
    &psupportVectors,
    &pclasses,
    &degree,
    &coef0,
    &gamma
    ))
  {

    dsp_arm_svm_polynomial_instance_f32Object *selfS = (dsp_arm_svm_polynomial_instance_f32Object *)S;
    GETARGUMENT(pdualCoefficients,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(psupportVectors,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pclasses,NPY_INT32,int32_t,int32_t);

    arm_svm_polynomial_init_f32(selfS->instance,
      nbOfSupportVectors,
      vectorDimension,
      intercept,
      pdualCoefficients_converted,
      psupportVectors_converted,
      pclasses_converted,
      degree,
      coef0,
      gamma
      );
    Py_RETURN_NONE;

  }
  return(NULL);
}

static PyObject *
cmsis_arm_svm_rbf_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pdualCoefficients=NULL; // input
  float32_t *pdualCoefficients_converted=NULL; // input
  PyObject *psupportVectors=NULL; // input
  float32_t *psupportVectors_converted=NULL; // input
  PyObject *pclasses=NULL; // input
  int32_t *pclasses_converted=NULL; // input
  uint32_t blockSize; // input

  uint32_t nbOfSupportVectors;
  uint32_t vectorDimension;
  float32_t intercept;
  float32_t       gamma;

  if (PyArg_ParseTuple(args,"OkkfOOOf",&S,
    &nbOfSupportVectors,
    &vectorDimension,
    &intercept,
    &pdualCoefficients,
    &psupportVectors,
    &pclasses,
    &gamma
    ))
  {

    dsp_arm_svm_rbf_instance_f32Object *selfS = (dsp_arm_svm_rbf_instance_f32Object *)S;
    GETARGUMENT(pdualCoefficients,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(psupportVectors,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pclasses,NPY_INT32,int32_t,int32_t);

    arm_svm_rbf_init_f32(selfS->instance,
      nbOfSupportVectors,
      vectorDimension,
      intercept,
      pdualCoefficients_converted,
      psupportVectors_converted,
      pclasses_converted,
      gamma
      );
    Py_RETURN_NONE;

  }
  return(NULL);
}

static PyObject *
cmsis_arm_svm_sigmoid_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pdualCoefficients=NULL; // input
  float32_t *pdualCoefficients_converted=NULL; // input
  PyObject *psupportVectors=NULL; // input
  float32_t *psupportVectors_converted=NULL; // input
  PyObject *pclasses=NULL; // input
  int32_t *pclasses_converted=NULL; // input
  uint32_t blockSize; // input

  uint32_t nbOfSupportVectors;
  uint32_t vectorDimension;
  float32_t intercept;
  float32_t       coef0;          
  float32_t       gamma;

  if (PyArg_ParseTuple(args,"OkkfOOOff",&S,
    &nbOfSupportVectors,
    &vectorDimension,
    &intercept,
    &pdualCoefficients,
    &psupportVectors,
    &pclasses,
    &coef0,
    &gamma
    ))
  {

    dsp_arm_svm_sigmoid_instance_f32Object *selfS = (dsp_arm_svm_sigmoid_instance_f32Object *)S;
    GETARGUMENT(pdualCoefficients,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(psupportVectors,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pclasses,NPY_INT32,int32_t,int32_t);

    arm_svm_sigmoid_init_f32(selfS->instance,
      nbOfSupportVectors,
      vectorDimension,
      intercept,
      pdualCoefficients_converted,
      psupportVectors_converted,
      pclasses_converted,
      coef0,
      gamma
      );
    Py_RETURN_NONE;

  }
  return(NULL);
}


#define SVMPREDICT(NAME)                                                                          \
static PyObject *                                                                                 \
cmsis_arm_svm_##NAME##_predict_f32(PyObject *obj, PyObject *args)                                 \
{                                                                                                 \
                                                                                                  \
  PyObject *S=NULL;                                                                               \
  PyObject *pSrc=NULL;                                                                            \
  float32_t *pSrc_converted=NULL;                                                                 \
  int32_t dst;                                                                                    \
  uint32_t blockSize;                                                                             \
                                                                                                  \
  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))                                                       \
  {                                                                                               \
                                                                                                  \
    dsp_arm_svm_##NAME##_instance_f32Object *selfS = (dsp_arm_svm_##NAME##_instance_f32Object *)S;\
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);                                                \
                                                                                                  \
                                                                                                  \
                                                                                                  \
    arm_svm_##NAME##_predict_f32(selfS->instance,pSrc_converted,&dst);                            \
    PyObject* resultOBJ=Py_BuildValue("i",dst);                                                   \
    PyObject *pythonResult = Py_BuildValue("O",resultOBJ);                                        \
                                                                                                  \
    FREEARGUMENT(pSrc_converted);                                                                 \
    Py_DECREF(resultOBJ);                                                                         \
    return(pythonResult);                                                                         \
                                                                                                  \
  }                                                                                               \
  return(NULL);                                                                                   \
}

SVMPREDICT(linear);
SVMPREDICT(polynomial);
SVMPREDICT(rbf);
SVMPREDICT(sigmoid);

static PyMethodDef CMSISDSPMethods[] = {


{"arm_svm_linear_init_f32",  cmsis_arm_svm_linear_init_f32, METH_VARARGS,""},
{"arm_svm_linear_predict_f32",  cmsis_arm_svm_linear_predict_f32, METH_VARARGS,""},

{"arm_svm_polynomial_init_f32",  cmsis_arm_svm_polynomial_init_f32, METH_VARARGS,""},
{"arm_svm_polynomial_predict_f32",  cmsis_arm_svm_polynomial_predict_f32, METH_VARARGS,""},

{"arm_svm_rbf_init_f32",  cmsis_arm_svm_rbf_init_f32, METH_VARARGS,""},
{"arm_svm_rbf_predict_f32",  cmsis_arm_svm_rbf_predict_f32, METH_VARARGS,""},

{"arm_svm_sigmoid_init_f32",  cmsis_arm_svm_sigmoid_init_f32, METH_VARARGS,""},
{"arm_svm_sigmoid_predict_f32",  cmsis_arm_svm_sigmoid_predict_f32, METH_VARARGS,""},

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