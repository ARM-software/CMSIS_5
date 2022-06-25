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

#define MODNAME "cmsisdsp_bayes"
#define MODINITNAME cmsisdsp_bayes

#include "cmsisdsp_module.h"



NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);



typedef struct {
    PyObject_HEAD
    arm_gaussian_naive_bayes_instance_f32 *instance;
} dsp_arm_gaussian_naive_bayes_instance_f32Object;


static void
arm_gaussian_naive_bayes_instance_f32_dealloc(dsp_arm_gaussian_naive_bayes_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {

       if (self->instance->theta)
       {
          PyMem_Free((float32_t*)self->instance->theta);
       }

       if (self->instance->sigma)
       {
          PyMem_Free((float32_t*)self->instance->sigma);
       }

       if (self->instance->classPriors)
       {
          PyMem_Free((float32_t*)self->instance->classPriors);
       }

       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_gaussian_naive_bayes_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_gaussian_naive_bayes_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_gaussian_naive_bayes_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_gaussian_naive_bayes_instance_f32));
        self->instance->theta=NULL;
        self->instance->sigma=NULL;
        self->instance->classPriors=NULL;

    }


    return (PyObject *)self;
}

static int
arm_gaussian_naive_bayes_instance_f32_init(dsp_arm_gaussian_naive_bayes_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

PyObject *theta=NULL;
PyObject *sigma=NULL;
PyObject *classPriors=NULL;

char *kwlist[] = {
"vectorDimension",
"numberOfClasses",
"theta",
"sigma",
"classPriors",
"epsilon",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|iiOOOf", kwlist,
  &self->instance->vectorDimension
,&self->instance->numberOfClasses
,&theta
,&sigma
,&classPriors
,&self->instance->epsilon
))
    {

      INITARRAYFIELD(theta,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(sigma,NPY_DOUBLE,double,float32_t);
      INITARRAYFIELD(classPriors,NPY_DOUBLE,double,float32_t);
    }
    return 0;
}

GETFIELD(arm_gaussian_naive_bayes_instance_f32,vectorDimension,"i");
GETFIELD(arm_gaussian_naive_bayes_instance_f32,numberOfClasses,"i");
GETFIELD(arm_gaussian_naive_bayes_instance_f32,epsilon,"f");


static PyMethodDef arm_gaussian_naive_bayes_instance_f32_methods[] = {

    {"vectorDimension", (PyCFunction) Method_arm_gaussian_naive_bayes_instance_f32_vectorDimension,METH_NOARGS,"vectorDimension"},
    {"numberOfClasses", (PyCFunction) Method_arm_gaussian_naive_bayes_instance_f32_numberOfClasses,METH_NOARGS,"numberOfClasses"},
    {"epsilon", (PyCFunction) Method_arm_gaussian_naive_bayes_instance_f32_epsilon,METH_NOARGS,"epsilon"},
   
    {NULL}  /* Sentinel */
};


DSPType(arm_gaussian_naive_bayes_instance_f32,arm_gaussian_naive_bayes_instance_f32_new,arm_gaussian_naive_bayes_instance_f32_dealloc,arm_gaussian_naive_bayes_instance_f32_init,arm_gaussian_naive_bayes_instance_f32_methods);




void typeRegistration(PyObject *module) {

  
  
  ADDTYPE(arm_gaussian_naive_bayes_instance_f32);
  
  
}


static PyObject *
cmsis_arm_gaussian_naive_bayes_predict_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t nbClasses; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_gaussian_naive_bayes_instance_f32Object *selfS = (dsp_arm_gaussian_naive_bayes_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);

    nbClasses=selfS->instance->numberOfClasses;


    pDst=PyMem_Malloc(sizeof(float32_t)*nbClasses);
    float32_t *temp=PyMem_Malloc(sizeof(float32_t)*nbClasses);


    uint32_t res=arm_gaussian_naive_bayes_predict_f32(selfS->instance,pSrc_converted,pDst,temp);
 FLOATARRAY1(pDstOBJ,nbClasses,pDst);

    PyObject *pythonResult = Py_BuildValue("Ok",pDstOBJ,res);

    FREEARGUMENT(pSrc_converted);
    PyMem_Free(temp);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyMethodDef CMSISDSPMethods[] = {



{"arm_gaussian_naive_bayes_predict_f32",  cmsis_arm_gaussian_naive_bayes_predict_f32, METH_VARARGS,""},



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