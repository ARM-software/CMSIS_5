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

#define MODNAME "cmsisdsp_controller"
#define MODINITNAME cmsisdsp_controller

#include "cmsisdsp_module.h"



NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);



typedef struct {
    PyObject_HEAD
    arm_pid_instance_q15 *instance;
} dsp_arm_pid_instance_q15Object;


static void
arm_pid_instance_q15_dealloc(dsp_arm_pid_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_pid_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_pid_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_pid_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_pid_instance_q15));


    }


    return (PyObject *)self;
}

static int
arm_pid_instance_q15_init(dsp_arm_pid_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

char *kwlist[] = {
"A0","A1","A2","state","Kp","Ki","Kd",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhhhhhh", kwlist,&self->instance->A0
,&self->instance->A1
,&self->instance->A2
,&self->instance->state
,&self->instance->Kp
,&self->instance->Ki
,&self->instance->Kd
))
    {


    }
    return 0;
}

GETFIELD(arm_pid_instance_q15,A0,"h");
GETFIELD(arm_pid_instance_q15,A1,"h");
GETFIELD(arm_pid_instance_q15,A2,"h");
GETFIELD(arm_pid_instance_q15,state,"h");
GETFIELD(arm_pid_instance_q15,Kp,"h");
GETFIELD(arm_pid_instance_q15,Ki,"h");
GETFIELD(arm_pid_instance_q15,Kd,"h");


static PyMethodDef arm_pid_instance_q15_methods[] = {

    {"A0", (PyCFunction) Method_arm_pid_instance_q15_A0,METH_NOARGS,"A0"},
    {"A1", (PyCFunction) Method_arm_pid_instance_q15_A1,METH_NOARGS,"A1"},
    {"A2", (PyCFunction) Method_arm_pid_instance_q15_A2,METH_NOARGS,"A2"},
    {"state", (PyCFunction) Method_arm_pid_instance_q15_state,METH_NOARGS,"state"},
    {"Kp", (PyCFunction) Method_arm_pid_instance_q15_Kp,METH_NOARGS,"Kp"},
    {"Ki", (PyCFunction) Method_arm_pid_instance_q15_Ki,METH_NOARGS,"Ki"},
    {"Kd", (PyCFunction) Method_arm_pid_instance_q15_Kd,METH_NOARGS,"Kd"},

    {NULL}  /* Sentinel */
};


DSPType(arm_pid_instance_q15,arm_pid_instance_q15_new,arm_pid_instance_q15_dealloc,arm_pid_instance_q15_init,arm_pid_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_pid_instance_q31 *instance;
} dsp_arm_pid_instance_q31Object;


static void
arm_pid_instance_q31_dealloc(dsp_arm_pid_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_pid_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_pid_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_pid_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_pid_instance_q31));


    }


    return (PyObject *)self;
}

static int
arm_pid_instance_q31_init(dsp_arm_pid_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

char *kwlist[] = {
"A0","A1","A2","state","Kp","Ki","Kd",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|iiiiiii", kwlist,&self->instance->A0
,&self->instance->A1
,&self->instance->A2
,&self->instance->state
,&self->instance->Kp
,&self->instance->Ki
,&self->instance->Kd
))
    {


    }
    return 0;
}

GETFIELD(arm_pid_instance_q31,A0,"i");
GETFIELD(arm_pid_instance_q31,A1,"i");
GETFIELD(arm_pid_instance_q31,A2,"i");
GETFIELD(arm_pid_instance_q31,state,"i");
GETFIELD(arm_pid_instance_q31,Kp,"i");
GETFIELD(arm_pid_instance_q31,Ki,"i");
GETFIELD(arm_pid_instance_q31,Kd,"i");


static PyMethodDef arm_pid_instance_q31_methods[] = {

    {"A0", (PyCFunction) Method_arm_pid_instance_q31_A0,METH_NOARGS,"A0"},
    {"A1", (PyCFunction) Method_arm_pid_instance_q31_A1,METH_NOARGS,"A1"},
    {"A2", (PyCFunction) Method_arm_pid_instance_q31_A2,METH_NOARGS,"A2"},
    {"state", (PyCFunction) Method_arm_pid_instance_q31_state,METH_NOARGS,"state"},
    {"Kp", (PyCFunction) Method_arm_pid_instance_q31_Kp,METH_NOARGS,"Kp"},
    {"Ki", (PyCFunction) Method_arm_pid_instance_q31_Ki,METH_NOARGS,"Ki"},
    {"Kd", (PyCFunction) Method_arm_pid_instance_q31_Kd,METH_NOARGS,"Kd"},

    {NULL}  /* Sentinel */
};


DSPType(arm_pid_instance_q31,arm_pid_instance_q31_new,arm_pid_instance_q31_dealloc,arm_pid_instance_q31_init,arm_pid_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_pid_instance_f32 *instance;
} dsp_arm_pid_instance_f32Object;


static void
arm_pid_instance_f32_dealloc(dsp_arm_pid_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_pid_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_pid_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_pid_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_pid_instance_f32));


    }


    return (PyObject *)self;
}

static int
arm_pid_instance_f32_init(dsp_arm_pid_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

char *kwlist[] = {
"A0","A1","A2","state","Kp","Ki","Kd",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|fffffff", kwlist,&self->instance->A0
,&self->instance->A1
,&self->instance->A2
,&self->instance->state
,&self->instance->Kp
,&self->instance->Ki
,&self->instance->Kd
))
    {


    }
    return 0;
}

GETFIELD(arm_pid_instance_f32,A0,"f");
GETFIELD(arm_pid_instance_f32,A1,"f");
GETFIELD(arm_pid_instance_f32,A2,"f");
GETFIELD(arm_pid_instance_f32,state,"f");
GETFIELD(arm_pid_instance_f32,Kp,"f");
GETFIELD(arm_pid_instance_f32,Ki,"f");
GETFIELD(arm_pid_instance_f32,Kd,"f");


static PyMethodDef arm_pid_instance_f32_methods[] = {

    {"A0", (PyCFunction) Method_arm_pid_instance_f32_A0,METH_NOARGS,"A0"},
    {"A1", (PyCFunction) Method_arm_pid_instance_f32_A1,METH_NOARGS,"A1"},
    {"A2", (PyCFunction) Method_arm_pid_instance_f32_A2,METH_NOARGS,"A2"},
    {"state", (PyCFunction) Method_arm_pid_instance_f32_state,METH_NOARGS,"state"},
    {"Kp", (PyCFunction) Method_arm_pid_instance_f32_Kp,METH_NOARGS,"Kp"},
    {"Ki", (PyCFunction) Method_arm_pid_instance_f32_Ki,METH_NOARGS,"Ki"},
    {"Kd", (PyCFunction) Method_arm_pid_instance_f32_Kd,METH_NOARGS,"Kd"},

    {NULL}  /* Sentinel */
};


DSPType(arm_pid_instance_f32,arm_pid_instance_f32_new,arm_pid_instance_f32_dealloc,arm_pid_instance_f32_init,arm_pid_instance_f32_methods);




void typeRegistration(PyObject *module) {

  
  
  ADDTYPE(arm_pid_instance_q15);
  ADDTYPE(arm_pid_instance_q31);
  ADDTYPE(arm_pid_instance_f32);
  
  
}


static PyObject *
cmsis_arm_pid_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  int32_t resetStateFlag; // input

  if (PyArg_ParseTuple(args,"Oi",&S,&resetStateFlag))
  {

    dsp_arm_pid_instance_f32Object *selfS = (dsp_arm_pid_instance_f32Object *)S;

    arm_pid_init_f32(selfS->instance,resetStateFlag);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_pid_reset_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input

  if (PyArg_ParseTuple(args,"O",&S))
  {

    dsp_arm_pid_instance_f32Object *selfS = (dsp_arm_pid_instance_f32Object *)S;

    arm_pid_reset_f32(selfS->instance);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_pid_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  int32_t resetStateFlag; // input

  if (PyArg_ParseTuple(args,"Oi",&S,&resetStateFlag))
  {

    dsp_arm_pid_instance_q31Object *selfS = (dsp_arm_pid_instance_q31Object *)S;

    arm_pid_init_q31(selfS->instance,resetStateFlag);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_pid_reset_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input

  if (PyArg_ParseTuple(args,"O",&S))
  {

    dsp_arm_pid_instance_q31Object *selfS = (dsp_arm_pid_instance_q31Object *)S;

    arm_pid_reset_q31(selfS->instance);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_pid_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  int32_t resetStateFlag; // input

  if (PyArg_ParseTuple(args,"Oi",&S,&resetStateFlag))
  {

    dsp_arm_pid_instance_q15Object *selfS = (dsp_arm_pid_instance_q15Object *)S;

    arm_pid_init_q15(selfS->instance,resetStateFlag);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_pid_reset_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input

  if (PyArg_ParseTuple(args,"O",&S))
  {

    dsp_arm_pid_instance_q15Object *selfS = (dsp_arm_pid_instance_q15Object *)S;

    arm_pid_reset_q15(selfS->instance);
    Py_RETURN_NONE;

  }
  return(NULL);
}





static PyObject *
cmsis_arm_sin_cos_f32(PyObject *obj, PyObject *args)
{

  float32_t theta; // input
  float32_t pS;
  float32_t pC;

  if (PyArg_ParseTuple(args,"f",&theta))
  {


    
    arm_sin_cos_f32(theta,&pS,&pC);
    
    PyObject* retS=Py_BuildValue("f",pS);
    PyObject* retC=Py_BuildValue("f",pC);

    PyObject *pythonResult = Py_BuildValue("OO",retS,retC);

    Py_DECREF(retS);
    Py_DECREF(retC);

    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_sin_cos_q31(PyObject *obj, PyObject *args)
{

  q31_t theta; // input
  q31_t pS;
  q31_t pC;

  if (PyArg_ParseTuple(args,"i",&theta))
  {

    arm_sin_cos_q31(theta,&pS,&pC);
    
    PyObject* retS=Py_BuildValue("i",pS);
    PyObject* retC=Py_BuildValue("i",pC);

    PyObject *pythonResult = Py_BuildValue("OO",retS,retC);

    Py_DECREF(retS);
    Py_DECREF(retC);

    return(pythonResult);

  }
  return(NULL);
}




static PyObject *
cmsis_arm_pid_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  float32_t in; // input

  if (PyArg_ParseTuple(args,"Of",&S,&in))
  {

    dsp_arm_pid_instance_f32Object *selfS = (dsp_arm_pid_instance_f32Object *)S;

    float32_t returnValue = arm_pid_f32(selfS->instance,in);
    PyObject* theReturnOBJ=Py_BuildValue("f",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_pid_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  q31_t in; // input

  if (PyArg_ParseTuple(args,"Oi",&S,&in))
  {

    dsp_arm_pid_instance_q31Object *selfS = (dsp_arm_pid_instance_q31Object *)S;

    q31_t returnValue = arm_pid_q31(selfS->instance,in);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_pid_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  q15_t in; // input

  if (PyArg_ParseTuple(args,"Oh",&S,&in))
  {

    dsp_arm_pid_instance_q15Object *selfS = (dsp_arm_pid_instance_q15Object *)S;

    q15_t returnValue = arm_pid_q15(selfS->instance,in);
    PyObject* theReturnOBJ=Py_BuildValue("h",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_arm_clarke_f32(PyObject *obj, PyObject *args)
{

  float32_t Ia; // input
  float32_t Ib; // input
  PyObject *pIalpha=NULL; // input
  float32_t *pIalpha_converted=NULL; // input
  PyObject *pIbeta=NULL; // input
  float32_t *pIbeta_converted=NULL; // input

  if (PyArg_ParseTuple(args,"ffOO",&Ia,&Ib,&pIalpha,&pIbeta))
  {

    GETARGUMENT(pIalpha,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pIbeta,NPY_DOUBLE,double,float32_t);

    arm_clarke_f32(Ia,Ib,pIalpha_converted,pIbeta_converted);
    FREEARGUMENT(pIalpha_converted);
    FREEARGUMENT(pIbeta_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_clarke_q31(PyObject *obj, PyObject *args)
{

  q31_t Ia; // input
  q31_t Ib; // input
  PyObject *pIalpha=NULL; // input
  q31_t *pIalpha_converted=NULL; // input
  PyObject *pIbeta=NULL; // input
  q31_t *pIbeta_converted=NULL; // input

  if (PyArg_ParseTuple(args,"iiOO",&Ia,&Ib,&pIalpha,&pIbeta))
  {

    GETARGUMENT(pIalpha,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pIbeta,NPY_INT32,int32_t,int32_t);

    arm_clarke_q31(Ia,Ib,pIalpha_converted,pIbeta_converted);
    FREEARGUMENT(pIalpha_converted);
    FREEARGUMENT(pIbeta_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}




static PyObject *
cmsis_arm_inv_clarke_f32(PyObject *obj, PyObject *args)
{

  float32_t Ialpha; // input
  float32_t Ibeta; // input
  PyObject *pIa=NULL; // input
  float32_t *pIa_converted=NULL; // input
  PyObject *pIb=NULL; // input
  float32_t *pIb_converted=NULL; // input

  if (PyArg_ParseTuple(args,"ffOO",&Ialpha,&Ibeta,&pIa,&pIb))
  {

    GETARGUMENT(pIa,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pIb,NPY_DOUBLE,double,float32_t);

    arm_inv_clarke_f32(Ialpha,Ibeta,pIa_converted,pIb_converted);
    FREEARGUMENT(pIa_converted);
    FREEARGUMENT(pIb_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_inv_clarke_q31(PyObject *obj, PyObject *args)
{

  q31_t Ialpha; // input
  q31_t Ibeta; // input
  PyObject *pIa=NULL; // input
  q31_t *pIa_converted=NULL; // input
  PyObject *pIb=NULL; // input
  q31_t *pIb_converted=NULL; // input

  if (PyArg_ParseTuple(args,"iiOO",&Ialpha,&Ibeta,&pIa,&pIb))
  {

    GETARGUMENT(pIa,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pIb,NPY_INT32,int32_t,int32_t);

    arm_inv_clarke_q31(Ialpha,Ibeta,pIa_converted,pIb_converted);
    FREEARGUMENT(pIa_converted);
    FREEARGUMENT(pIb_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}




static PyObject *
cmsis_arm_park_f32(PyObject *obj, PyObject *args)
{

  float32_t Ialpha; // input
  float32_t Ibeta; // input
  PyObject *pId=NULL; // input
  float32_t *pId_converted=NULL; // input
  PyObject *pIq=NULL; // input
  float32_t *pIq_converted=NULL; // input
  float32_t sinVal; // input
  float32_t cosVal; // input

  if (PyArg_ParseTuple(args,"ffOOff",&Ialpha,&Ibeta,&pId,&pIq,&sinVal,&cosVal))
  {

    GETARGUMENT(pId,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pIq,NPY_DOUBLE,double,float32_t);

    arm_park_f32(Ialpha,Ibeta,pId_converted,pIq_converted,sinVal,cosVal);
    FREEARGUMENT(pId_converted);
    FREEARGUMENT(pIq_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_park_q31(PyObject *obj, PyObject *args)
{

  q31_t Ialpha; // input
  q31_t Ibeta; // input
  PyObject *pId=NULL; // input
  q31_t *pId_converted=NULL; // input
  PyObject *pIq=NULL; // input
  q31_t *pIq_converted=NULL; // input
  q31_t sinVal; // input
  q31_t cosVal; // input

  if (PyArg_ParseTuple(args,"iiOOii",&Ialpha,&Ibeta,&pId,&pIq,&sinVal,&cosVal))
  {

    GETARGUMENT(pId,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pIq,NPY_INT32,int32_t,int32_t);

    arm_park_q31(Ialpha,Ibeta,pId_converted,pIq_converted,sinVal,cosVal);
    FREEARGUMENT(pId_converted);
    FREEARGUMENT(pIq_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}





static PyObject *
cmsis_arm_inv_park_f32(PyObject *obj, PyObject *args)
{

  float32_t Id; // input
  float32_t Iq; // input
  PyObject *pIalpha=NULL; // input
  float32_t *pIalpha_converted=NULL; // input
  PyObject *pIbeta=NULL; // input
  float32_t *pIbeta_converted=NULL; // input
  float32_t sinVal; // input
  float32_t cosVal; // input

  if (PyArg_ParseTuple(args,"ffOOff",&Id,&Iq,&pIalpha,&pIbeta,&sinVal,&cosVal))
  {

    GETARGUMENT(pIalpha,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pIbeta,NPY_DOUBLE,double,float32_t);

    arm_inv_park_f32(Id,Iq,pIalpha_converted,pIbeta_converted,sinVal,cosVal);
    FREEARGUMENT(pIalpha_converted);
    FREEARGUMENT(pIbeta_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_inv_park_q31(PyObject *obj, PyObject *args)
{

  q31_t Id; // input
  q31_t Iq; // input
  PyObject *pIalpha=NULL; // input
  q31_t *pIalpha_converted=NULL; // input
  PyObject *pIbeta=NULL; // input
  q31_t *pIbeta_converted=NULL; // input
  q31_t sinVal; // input
  q31_t cosVal; // input

  if (PyArg_ParseTuple(args,"iiOOii",&Id,&Iq,&pIalpha,&pIbeta,&sinVal,&cosVal))
  {

    GETARGUMENT(pIalpha,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pIbeta,NPY_INT32,int32_t,int32_t);

    arm_inv_park_q31(Id,Iq,pIalpha_converted,pIbeta_converted,sinVal,cosVal);
    FREEARGUMENT(pIalpha_converted);
    FREEARGUMENT(pIbeta_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}




static PyMethodDef CMSISDSPMethods[] = {



{"arm_pid_init_f32",  cmsis_arm_pid_init_f32, METH_VARARGS,""},
{"arm_pid_reset_f32",  cmsis_arm_pid_reset_f32, METH_VARARGS,""},
{"arm_pid_init_q31",  cmsis_arm_pid_init_q31, METH_VARARGS,""},
{"arm_pid_reset_q31",  cmsis_arm_pid_reset_q31, METH_VARARGS,""},
{"arm_pid_init_q15",  cmsis_arm_pid_init_q15, METH_VARARGS,""},
{"arm_pid_reset_q15",  cmsis_arm_pid_reset_q15, METH_VARARGS,""},


{"arm_sin_cos_f32",  cmsis_arm_sin_cos_f32, METH_VARARGS,""},
{"arm_sin_cos_q31",  cmsis_arm_sin_cos_q31, METH_VARARGS,""},

{"arm_pid_f32",  cmsis_arm_pid_f32, METH_VARARGS,""},
{"arm_pid_q31",  cmsis_arm_pid_q31, METH_VARARGS,""},
{"arm_pid_q15",  cmsis_arm_pid_q15, METH_VARARGS,""},

{"arm_clarke_f32",  cmsis_arm_clarke_f32, METH_VARARGS,""},
{"arm_clarke_q31",  cmsis_arm_clarke_q31, METH_VARARGS,""},
{"arm_inv_clarke_f32",  cmsis_arm_inv_clarke_f32, METH_VARARGS,""},
{"arm_inv_clarke_q31",  cmsis_arm_inv_clarke_q31, METH_VARARGS,""},
{"arm_park_f32",  cmsis_arm_park_f32, METH_VARARGS,""},
{"arm_park_q31",  cmsis_arm_park_q31, METH_VARARGS,""},
{"arm_inv_park_f32",  cmsis_arm_inv_park_f32, METH_VARARGS,""},
{"arm_inv_park_q31",  cmsis_arm_inv_park_q31, METH_VARARGS,""},


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