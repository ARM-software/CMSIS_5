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

#define MODNAME "cmsisdsp_filtering"
#define MODINITNAME cmsisdsp_filtering

#include "cmsisdsp_module.h"

NUMPYVECTORFROMBUFFER(f32,float32_t,NPY_FLOAT);

typedef struct {
    PyObject_HEAD
    arm_fir_instance_q7 *instance;
} dsp_arm_fir_instance_q7Object;


static void
arm_fir_instance_q7_dealloc(dsp_arm_fir_instance_q7Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q7_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_instance_q7_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_instance_q7Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_instance_q7Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_instance_q7));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_instance_q7_init(dsp_arm_fir_instance_q7Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numTaps
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_instance_q7,numTaps,"h");


static PyMethodDef arm_fir_instance_q7_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_instance_q7_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_instance_q7,arm_fir_instance_q7_new,arm_fir_instance_q7_dealloc,arm_fir_instance_q7_init,arm_fir_instance_q7_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_instance_q15 *instance;
} dsp_arm_fir_instance_q15Object;


static void
arm_fir_instance_q15_dealloc(dsp_arm_fir_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_instance_q15));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_instance_q15_init(dsp_arm_fir_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numTaps
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_instance_q15,numTaps,"h");


static PyMethodDef arm_fir_instance_q15_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_instance_q15_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_instance_q15,arm_fir_instance_q15_new,arm_fir_instance_q15_dealloc,arm_fir_instance_q15_init,arm_fir_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_instance_q31 *instance;
} dsp_arm_fir_instance_q31Object;


static void
arm_fir_instance_q31_dealloc(dsp_arm_fir_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_instance_q31));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_instance_q31_init(dsp_arm_fir_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numTaps
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_instance_q31,numTaps,"h");


static PyMethodDef arm_fir_instance_q31_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_instance_q31_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_instance_q31,arm_fir_instance_q31_new,arm_fir_instance_q31_dealloc,arm_fir_instance_q31_init,arm_fir_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_instance_f32 *instance;
} dsp_arm_fir_instance_f32Object;

typedef struct {
    PyObject_HEAD
    arm_fir_instance_f64 *instance;
} dsp_arm_fir_instance_f64Object;

static void
arm_fir_instance_f32_dealloc(dsp_arm_fir_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static void
arm_fir_instance_f64_dealloc(dsp_arm_fir_instance_f64Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float64_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_instance_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static PyObject *
arm_fir_instance_f64_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_instance_f64Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_instance_f64Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_instance_f64));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_instance_f32_init(dsp_arm_fir_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numTaps
))
    {


    }
    return 0;
}

static int
arm_fir_instance_f64_init(dsp_arm_fir_instance_f64Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numTaps
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_instance_f32,numTaps,"h");
GETFIELD(arm_fir_instance_f64,numTaps,"h");


static PyMethodDef arm_fir_instance_f32_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_instance_f32_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};

static PyMethodDef arm_fir_instance_f64_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_instance_f64_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_instance_f32,arm_fir_instance_f32_new,arm_fir_instance_f32_dealloc,arm_fir_instance_f32_init,arm_fir_instance_f32_methods);
DSPType(arm_fir_instance_f64,arm_fir_instance_f64_new,arm_fir_instance_f64_dealloc,arm_fir_instance_f64_init,arm_fir_instance_f64_methods);


typedef struct {
    PyObject_HEAD
    arm_biquad_casd_df1_inst_q15 *instance;
} dsp_arm_biquad_casd_df1_inst_q15Object;


static void
arm_biquad_casd_df1_inst_q15_dealloc(dsp_arm_biquad_casd_df1_inst_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_biquad_casd_df1_inst_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_biquad_casd_df1_inst_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_biquad_casd_df1_inst_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_biquad_casd_df1_inst_q15));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_biquad_casd_df1_inst_q15_init(dsp_arm_biquad_casd_df1_inst_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages","postShift",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,&self->instance->numStages
,&self->instance->postShift
))
    {


    }
    return 0;
}

GETFIELD(arm_biquad_casd_df1_inst_q15,numStages,"i");
GETFIELD(arm_biquad_casd_df1_inst_q15,postShift,"i");


static PyMethodDef arm_biquad_casd_df1_inst_q15_methods[] = {

    {"numStages", (PyCFunction) Method_arm_biquad_casd_df1_inst_q15_numStages,METH_NOARGS,"numStages"},
    {"postShift", (PyCFunction) Method_arm_biquad_casd_df1_inst_q15_postShift,METH_NOARGS,"postShift"},

    {NULL}  /* Sentinel */
};


DSPType(arm_biquad_casd_df1_inst_q15,arm_biquad_casd_df1_inst_q15_new,arm_biquad_casd_df1_inst_q15_dealloc,arm_biquad_casd_df1_inst_q15_init,arm_biquad_casd_df1_inst_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_biquad_casd_df1_inst_q31 *instance;
} dsp_arm_biquad_casd_df1_inst_q31Object;


static void
arm_biquad_casd_df1_inst_q31_dealloc(dsp_arm_biquad_casd_df1_inst_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_biquad_casd_df1_inst_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_biquad_casd_df1_inst_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_biquad_casd_df1_inst_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_biquad_casd_df1_inst_q31));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_biquad_casd_df1_inst_q31_init(dsp_arm_biquad_casd_df1_inst_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages","postShift",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,&self->instance->numStages
,&self->instance->postShift
))
    {


    }
    return 0;
}

GETFIELD(arm_biquad_casd_df1_inst_q31,numStages,"i");
GETFIELD(arm_biquad_casd_df1_inst_q31,postShift,"i");


static PyMethodDef arm_biquad_casd_df1_inst_q31_methods[] = {

    {"numStages", (PyCFunction) Method_arm_biquad_casd_df1_inst_q31_numStages,METH_NOARGS,"numStages"},
    {"postShift", (PyCFunction) Method_arm_biquad_casd_df1_inst_q31_postShift,METH_NOARGS,"postShift"},

    {NULL}  /* Sentinel */
};


DSPType(arm_biquad_casd_df1_inst_q31,arm_biquad_casd_df1_inst_q31_new,arm_biquad_casd_df1_inst_q31_dealloc,arm_biquad_casd_df1_inst_q31_init,arm_biquad_casd_df1_inst_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_biquad_casd_df1_inst_f32 *instance;
} dsp_arm_biquad_casd_df1_inst_f32Object;


static void
arm_biquad_casd_df1_inst_f32_dealloc(dsp_arm_biquad_casd_df1_inst_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_biquad_casd_df1_inst_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_biquad_casd_df1_inst_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_biquad_casd_df1_inst_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_biquad_casd_df1_inst_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_biquad_casd_df1_inst_f32_init(dsp_arm_biquad_casd_df1_inst_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,&self->instance->numStages
))
    {


    }
    return 0;
}

GETFIELD(arm_biquad_casd_df1_inst_f32,numStages,"i");


static PyMethodDef arm_biquad_casd_df1_inst_f32_methods[] = {

    {"numStages", (PyCFunction) Method_arm_biquad_casd_df1_inst_f32_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_biquad_casd_df1_inst_f32,arm_biquad_casd_df1_inst_f32_new,arm_biquad_casd_df1_inst_f32_dealloc,arm_biquad_casd_df1_inst_f32_init,arm_biquad_casd_df1_inst_f32_methods);







typedef struct {
    PyObject_HEAD
    arm_fir_decimate_instance_q15 *instance;
} dsp_arm_fir_decimate_instance_q15Object;


static void
arm_fir_decimate_instance_q15_dealloc(dsp_arm_fir_decimate_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_decimate_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_decimate_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_decimate_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_decimate_instance_q15));

        self->instance->pCoeffs = NULL;
        self->instance->pState = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_decimate_instance_q15_init(dsp_arm_fir_decimate_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pCoeffs=NULL;
    PyObject *pState=NULL;
char *kwlist[] = {
"M","numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ih", kwlist,&self->instance->M
,&self->instance->numTaps
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_decimate_instance_q15,M,"i");
GETFIELD(arm_fir_decimate_instance_q15,numTaps,"h");


static PyMethodDef arm_fir_decimate_instance_q15_methods[] = {

    {"M", (PyCFunction) Method_arm_fir_decimate_instance_q15_M,METH_NOARGS,"M"},
    {"numTaps", (PyCFunction) Method_arm_fir_decimate_instance_q15_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_decimate_instance_q15,arm_fir_decimate_instance_q15_new,arm_fir_decimate_instance_q15_dealloc,arm_fir_decimate_instance_q15_init,arm_fir_decimate_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_decimate_instance_q31 *instance;
} dsp_arm_fir_decimate_instance_q31Object;


static void
arm_fir_decimate_instance_q31_dealloc(dsp_arm_fir_decimate_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_decimate_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_decimate_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_decimate_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_decimate_instance_q31));

        self->instance->pCoeffs = NULL;
        self->instance->pState = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_decimate_instance_q31_init(dsp_arm_fir_decimate_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pCoeffs=NULL;
    PyObject *pState=NULL;
char *kwlist[] = {
"M","numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ih", kwlist,&self->instance->M
,&self->instance->numTaps
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_decimate_instance_q31,M,"i");
GETFIELD(arm_fir_decimate_instance_q31,numTaps,"h");


static PyMethodDef arm_fir_decimate_instance_q31_methods[] = {

    {"M", (PyCFunction) Method_arm_fir_decimate_instance_q31_M,METH_NOARGS,"M"},
    {"numTaps", (PyCFunction) Method_arm_fir_decimate_instance_q31_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_decimate_instance_q31,arm_fir_decimate_instance_q31_new,arm_fir_decimate_instance_q31_dealloc,arm_fir_decimate_instance_q31_init,arm_fir_decimate_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_decimate_instance_f32 *instance;
} dsp_arm_fir_decimate_instance_f32Object;


static void
arm_fir_decimate_instance_f32_dealloc(dsp_arm_fir_decimate_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_decimate_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_decimate_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_decimate_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_decimate_instance_f32));

        self->instance->pCoeffs = NULL;
        self->instance->pState = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_decimate_instance_f32_init(dsp_arm_fir_decimate_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pCoeffs=NULL;
    PyObject *pState=NULL;
char *kwlist[] = {
"M","numTaps",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ih", kwlist,&self->instance->M
,&self->instance->numTaps
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_decimate_instance_f32,M,"i");
GETFIELD(arm_fir_decimate_instance_f32,numTaps,"h");


static PyMethodDef arm_fir_decimate_instance_f32_methods[] = {

    {"M", (PyCFunction) Method_arm_fir_decimate_instance_f32_M,METH_NOARGS,"M"},
    {"numTaps", (PyCFunction) Method_arm_fir_decimate_instance_f32_numTaps,METH_NOARGS,"numTaps"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_decimate_instance_f32,arm_fir_decimate_instance_f32_new,arm_fir_decimate_instance_f32_dealloc,arm_fir_decimate_instance_f32_init,arm_fir_decimate_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_interpolate_instance_q15 *instance;
} dsp_arm_fir_interpolate_instance_q15Object;


static void
arm_fir_interpolate_instance_q15_dealloc(dsp_arm_fir_interpolate_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_interpolate_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_interpolate_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_interpolate_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_interpolate_instance_q15));

        self->instance->pCoeffs = NULL;
        self->instance->pState = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_interpolate_instance_q15_init(dsp_arm_fir_interpolate_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pCoeffs=NULL;
    PyObject *pState=NULL;
char *kwlist[] = {
"L","phaseLength",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ih", kwlist,&self->instance->L
,&self->instance->phaseLength
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_interpolate_instance_q15,L,"i");
GETFIELD(arm_fir_interpolate_instance_q15,phaseLength,"h");


static PyMethodDef arm_fir_interpolate_instance_q15_methods[] = {

    {"L", (PyCFunction) Method_arm_fir_interpolate_instance_q15_L,METH_NOARGS,"L"},
    {"phaseLength", (PyCFunction) Method_arm_fir_interpolate_instance_q15_phaseLength,METH_NOARGS,"phaseLength"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_interpolate_instance_q15,arm_fir_interpolate_instance_q15_new,arm_fir_interpolate_instance_q15_dealloc,arm_fir_interpolate_instance_q15_init,arm_fir_interpolate_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_interpolate_instance_q31 *instance;
} dsp_arm_fir_interpolate_instance_q31Object;


static void
arm_fir_interpolate_instance_q31_dealloc(dsp_arm_fir_interpolate_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_interpolate_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_interpolate_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_interpolate_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_interpolate_instance_q31));

        self->instance->pCoeffs = NULL;
        self->instance->pState = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_interpolate_instance_q31_init(dsp_arm_fir_interpolate_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pCoeffs=NULL;
    PyObject *pState=NULL;
char *kwlist[] = {
"L","phaseLength",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ih", kwlist,&self->instance->L
,&self->instance->phaseLength
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_interpolate_instance_q31,L,"i");
GETFIELD(arm_fir_interpolate_instance_q31,phaseLength,"h");


static PyMethodDef arm_fir_interpolate_instance_q31_methods[] = {

    {"L", (PyCFunction) Method_arm_fir_interpolate_instance_q31_L,METH_NOARGS,"L"},
    {"phaseLength", (PyCFunction) Method_arm_fir_interpolate_instance_q31_phaseLength,METH_NOARGS,"phaseLength"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_interpolate_instance_q31,arm_fir_interpolate_instance_q31_new,arm_fir_interpolate_instance_q31_dealloc,arm_fir_interpolate_instance_q31_init,arm_fir_interpolate_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_interpolate_instance_f32 *instance;
} dsp_arm_fir_interpolate_instance_f32Object;


static void
arm_fir_interpolate_instance_f32_dealloc(dsp_arm_fir_interpolate_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_interpolate_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_interpolate_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_interpolate_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_interpolate_instance_f32));

        self->instance->pCoeffs = NULL;
        self->instance->pState = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_interpolate_instance_f32_init(dsp_arm_fir_interpolate_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pCoeffs=NULL;
    PyObject *pState=NULL;
char *kwlist[] = {
"L","phaseLength",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ih", kwlist,&self->instance->L
,&self->instance->phaseLength
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_interpolate_instance_f32,L,"i");
GETFIELD(arm_fir_interpolate_instance_f32,phaseLength,"h");


static PyMethodDef arm_fir_interpolate_instance_f32_methods[] = {

    {"L", (PyCFunction) Method_arm_fir_interpolate_instance_f32_L,METH_NOARGS,"L"},
    {"phaseLength", (PyCFunction) Method_arm_fir_interpolate_instance_f32_phaseLength,METH_NOARGS,"phaseLength"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_interpolate_instance_f32,arm_fir_interpolate_instance_f32_new,arm_fir_interpolate_instance_f32_dealloc,arm_fir_interpolate_instance_f32_init,arm_fir_interpolate_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_biquad_cas_df1_32x64_ins_q31 *instance;
} dsp_arm_biquad_cas_df1_32x64_ins_q31Object;


static void
arm_biquad_cas_df1_32x64_ins_q31_dealloc(dsp_arm_biquad_cas_df1_32x64_ins_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_biquad_cas_df1_32x64_ins_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_biquad_cas_df1_32x64_ins_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_biquad_cas_df1_32x64_ins_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_biquad_cas_df1_32x64_ins_q31));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_biquad_cas_df1_32x64_ins_q31_init(dsp_arm_biquad_cas_df1_32x64_ins_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages","postShift",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,&self->instance->numStages
,&self->instance->postShift
))
    {


    }
    return 0;
}

GETFIELD(arm_biquad_cas_df1_32x64_ins_q31,numStages,"i");
GETFIELD(arm_biquad_cas_df1_32x64_ins_q31,postShift,"i");


static PyMethodDef arm_biquad_cas_df1_32x64_ins_q31_methods[] = {

    {"numStages", (PyCFunction) Method_arm_biquad_cas_df1_32x64_ins_q31_numStages,METH_NOARGS,"numStages"},
    {"postShift", (PyCFunction) Method_arm_biquad_cas_df1_32x64_ins_q31_postShift,METH_NOARGS,"postShift"},

    {NULL}  /* Sentinel */
};


DSPType(arm_biquad_cas_df1_32x64_ins_q31,arm_biquad_cas_df1_32x64_ins_q31_new,arm_biquad_cas_df1_32x64_ins_q31_dealloc,arm_biquad_cas_df1_32x64_ins_q31_init,arm_biquad_cas_df1_32x64_ins_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_biquad_cascade_df2T_instance_f32 *instance;
} dsp_arm_biquad_cascade_df2T_instance_f32Object;


static void
arm_biquad_cascade_df2T_instance_f32_dealloc(dsp_arm_biquad_cascade_df2T_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_biquad_cascade_df2T_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_biquad_cascade_df2T_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_biquad_cascade_df2T_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_biquad_cascade_df2T_instance_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_biquad_cascade_df2T_instance_f32_init(dsp_arm_biquad_cascade_df2T_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,&self->instance->numStages
))
    {


    }
    return 0;
}

GETFIELD(arm_biquad_cascade_df2T_instance_f32,numStages,"i");

static PyObject *                                                             
Method_arm_biquad_cascade_df2T_instance_f32_state(dsp_arm_biquad_cascade_df2T_instance_f32Object *self, PyObject *ignored)
{                
    float32_t *state=self->instance->pState;
    return(NumpyVectorFromf32Buffer(state,self->instance->numStages * 2));                                                  
} 

static PyMethodDef arm_biquad_cascade_df2T_instance_f32_methods[] = {

    {"numStages", (PyCFunction) Method_arm_biquad_cascade_df2T_instance_f32_numStages,METH_NOARGS,"numStages"},
    {"state", (PyCFunction) Method_arm_biquad_cascade_df2T_instance_f32_state,METH_NOARGS,"state"},
    {NULL}  /* Sentinel */
};


DSPType(arm_biquad_cascade_df2T_instance_f32,arm_biquad_cascade_df2T_instance_f32_new,arm_biquad_cascade_df2T_instance_f32_dealloc,arm_biquad_cascade_df2T_instance_f32_init,arm_biquad_cascade_df2T_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_biquad_cascade_stereo_df2T_instance_f32 *instance;
} dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object;


static void
arm_biquad_cascade_stereo_df2T_instance_f32_dealloc(dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_biquad_cascade_stereo_df2T_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_biquad_cascade_stereo_df2T_instance_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_biquad_cascade_stereo_df2T_instance_f32_init(dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,&self->instance->numStages
))
    {


    }
    return 0;
}

GETFIELD(arm_biquad_cascade_stereo_df2T_instance_f32,numStages,"i");


static PyMethodDef arm_biquad_cascade_stereo_df2T_instance_f32_methods[] = {

    {"numStages", (PyCFunction) Method_arm_biquad_cascade_stereo_df2T_instance_f32_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_biquad_cascade_stereo_df2T_instance_f32,arm_biquad_cascade_stereo_df2T_instance_f32_new,arm_biquad_cascade_stereo_df2T_instance_f32_dealloc,arm_biquad_cascade_stereo_df2T_instance_f32_init,arm_biquad_cascade_stereo_df2T_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_biquad_cascade_df2T_instance_f64 *instance;
} dsp_arm_biquad_cascade_df2T_instance_f64Object;


static void
arm_biquad_cascade_df2T_instance_f64_dealloc(dsp_arm_biquad_cascade_df2T_instance_f64Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float64_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_biquad_cascade_df2T_instance_f64_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_biquad_cascade_df2T_instance_f64Object *self;
    //printf("New called\n");

    self = (dsp_arm_biquad_cascade_df2T_instance_f64Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_biquad_cascade_df2T_instance_f64));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_biquad_cascade_df2T_instance_f64_init(dsp_arm_biquad_cascade_df2T_instance_f64Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,&self->instance->numStages
))
    {


    }
    return 0;
}

GETFIELD(arm_biquad_cascade_df2T_instance_f64,numStages,"i");


static PyMethodDef arm_biquad_cascade_df2T_instance_f64_methods[] = {

    {"numStages", (PyCFunction) Method_arm_biquad_cascade_df2T_instance_f64_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_biquad_cascade_df2T_instance_f64,arm_biquad_cascade_df2T_instance_f64_new,arm_biquad_cascade_df2T_instance_f64_dealloc,arm_biquad_cascade_df2T_instance_f64_init,arm_biquad_cascade_df2T_instance_f64_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_lattice_instance_q15 *instance;
} dsp_arm_fir_lattice_instance_q15Object;


static void
arm_fir_lattice_instance_q15_dealloc(dsp_arm_fir_lattice_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_lattice_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_lattice_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_lattice_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_lattice_instance_q15));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_lattice_instance_q15_init(dsp_arm_fir_lattice_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numStages
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_lattice_instance_q15,numStages,"h");


static PyMethodDef arm_fir_lattice_instance_q15_methods[] = {

    {"numStages", (PyCFunction) Method_arm_fir_lattice_instance_q15_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_lattice_instance_q15,arm_fir_lattice_instance_q15_new,arm_fir_lattice_instance_q15_dealloc,arm_fir_lattice_instance_q15_init,arm_fir_lattice_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_lattice_instance_q31 *instance;
} dsp_arm_fir_lattice_instance_q31Object;


static void
arm_fir_lattice_instance_q31_dealloc(dsp_arm_fir_lattice_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_lattice_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_lattice_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_lattice_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_lattice_instance_q31));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_lattice_instance_q31_init(dsp_arm_fir_lattice_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numStages
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_lattice_instance_q31,numStages,"h");


static PyMethodDef arm_fir_lattice_instance_q31_methods[] = {

    {"numStages", (PyCFunction) Method_arm_fir_lattice_instance_q31_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_lattice_instance_q31,arm_fir_lattice_instance_q31_new,arm_fir_lattice_instance_q31_dealloc,arm_fir_lattice_instance_q31_init,arm_fir_lattice_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_lattice_instance_f32 *instance;
} dsp_arm_fir_lattice_instance_f32Object;


static void
arm_fir_lattice_instance_f32_dealloc(dsp_arm_fir_lattice_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_lattice_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_lattice_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_lattice_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_lattice_instance_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_lattice_instance_f32_init(dsp_arm_fir_lattice_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numStages",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|h", kwlist,&self->instance->numStages
))
    {


    }
    return 0;
}

GETFIELD(arm_fir_lattice_instance_f32,numStages,"h");


static PyMethodDef arm_fir_lattice_instance_f32_methods[] = {

    {"numStages", (PyCFunction) Method_arm_fir_lattice_instance_f32_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_lattice_instance_f32,arm_fir_lattice_instance_f32_new,arm_fir_lattice_instance_f32_dealloc,arm_fir_lattice_instance_f32_init,arm_fir_lattice_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_iir_lattice_instance_q15 *instance;
} dsp_arm_iir_lattice_instance_q15Object;


static void
arm_iir_lattice_instance_q15_dealloc(dsp_arm_iir_lattice_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pkCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pkCoeffs);
       }


       if (self->instance->pvCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pvCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_iir_lattice_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_iir_lattice_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_iir_lattice_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_iir_lattice_instance_q15));

        self->instance->pState = NULL;
        self->instance->pkCoeffs = NULL;
        self->instance->pvCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_iir_lattice_instance_q15_init(dsp_arm_iir_lattice_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pkCoeffs=NULL;
    PyObject *pvCoeffs=NULL;
char *kwlist[] = {
"numStages","pkCoeffs","pvCoeffs",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hOO", kwlist,&self->instance->numStages
,&pkCoeffs
,&pvCoeffs
))
    {

    INITARRAYFIELD(pkCoeffs,NPY_INT16,int16_t,int16_t);
    INITARRAYFIELD(pvCoeffs,NPY_INT16,int16_t,int16_t);

    }
    return 0;
}

GETFIELD(arm_iir_lattice_instance_q15,numStages,"h");


static PyMethodDef arm_iir_lattice_instance_q15_methods[] = {

    {"numStages", (PyCFunction) Method_arm_iir_lattice_instance_q15_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_iir_lattice_instance_q15,arm_iir_lattice_instance_q15_new,arm_iir_lattice_instance_q15_dealloc,arm_iir_lattice_instance_q15_init,arm_iir_lattice_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_iir_lattice_instance_q31 *instance;
} dsp_arm_iir_lattice_instance_q31Object;


static void
arm_iir_lattice_instance_q31_dealloc(dsp_arm_iir_lattice_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pkCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pkCoeffs);
       }


       if (self->instance->pvCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pvCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_iir_lattice_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_iir_lattice_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_iir_lattice_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_iir_lattice_instance_q31));

        self->instance->pState = NULL;
        self->instance->pkCoeffs = NULL;
        self->instance->pvCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_iir_lattice_instance_q31_init(dsp_arm_iir_lattice_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pkCoeffs=NULL;
    PyObject *pvCoeffs=NULL;
char *kwlist[] = {
"numStages","pkCoeffs","pvCoeffs",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hOO", kwlist,&self->instance->numStages
,&pkCoeffs
,&pvCoeffs
))
    {

    INITARRAYFIELD(pkCoeffs,NPY_INT32,int32_t,int32_t);
    INITARRAYFIELD(pvCoeffs,NPY_INT32,int32_t,int32_t);

    }
    return 0;
}

GETFIELD(arm_iir_lattice_instance_q31,numStages,"h");


static PyMethodDef arm_iir_lattice_instance_q31_methods[] = {

    {"numStages", (PyCFunction) Method_arm_iir_lattice_instance_q31_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_iir_lattice_instance_q31,arm_iir_lattice_instance_q31_new,arm_iir_lattice_instance_q31_dealloc,arm_iir_lattice_instance_q31_init,arm_iir_lattice_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_iir_lattice_instance_f32 *instance;
} dsp_arm_iir_lattice_instance_f32Object;


static void
arm_iir_lattice_instance_f32_dealloc(dsp_arm_iir_lattice_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pkCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pkCoeffs);
       }


       if (self->instance->pvCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pvCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_iir_lattice_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_iir_lattice_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_iir_lattice_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_iir_lattice_instance_f32));

        self->instance->pState = NULL;
        self->instance->pkCoeffs = NULL;
        self->instance->pvCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_iir_lattice_instance_f32_init(dsp_arm_iir_lattice_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pkCoeffs=NULL;
    PyObject *pvCoeffs=NULL;
char *kwlist[] = {
"numStages","pkCoeffs","pvCoeffs",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hOO", kwlist,&self->instance->numStages
,&pkCoeffs
,&pvCoeffs
))
    {

    INITARRAYFIELD(pkCoeffs,NPY_DOUBLE,double,float32_t);
    INITARRAYFIELD(pvCoeffs,NPY_DOUBLE,double,float32_t);

    }
    return 0;
}

GETFIELD(arm_iir_lattice_instance_f32,numStages,"h");


static PyMethodDef arm_iir_lattice_instance_f32_methods[] = {

    {"numStages", (PyCFunction) Method_arm_iir_lattice_instance_f32_numStages,METH_NOARGS,"numStages"},

    {NULL}  /* Sentinel */
};


DSPType(arm_iir_lattice_instance_f32,arm_iir_lattice_instance_f32_new,arm_iir_lattice_instance_f32_dealloc,arm_iir_lattice_instance_f32_init,arm_iir_lattice_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_lms_instance_f32 *instance;
} dsp_arm_lms_instance_f32Object;


static void
arm_lms_instance_f32_dealloc(dsp_arm_lms_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_lms_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_lms_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_lms_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_lms_instance_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_lms_instance_f32_init(dsp_arm_lms_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps","mu",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hf", kwlist,&self->instance->numTaps
,&self->instance->mu
))
    {


    }
    return 0;
}

GETFIELD(arm_lms_instance_f32,numTaps,"h");
GETFIELD(arm_lms_instance_f32,mu,"f");


static PyMethodDef arm_lms_instance_f32_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_lms_instance_f32_numTaps,METH_NOARGS,"numTaps"},
    {"mu", (PyCFunction) Method_arm_lms_instance_f32_mu,METH_NOARGS,"mu"},

    {NULL}  /* Sentinel */
};


DSPType(arm_lms_instance_f32,arm_lms_instance_f32_new,arm_lms_instance_f32_dealloc,arm_lms_instance_f32_init,arm_lms_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_lms_instance_q15 *instance;
} dsp_arm_lms_instance_q15Object;


static void
arm_lms_instance_q15_dealloc(dsp_arm_lms_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_lms_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_lms_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_lms_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_lms_instance_q15));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_lms_instance_q15_init(dsp_arm_lms_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps","mu","postShift",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhi", kwlist,&self->instance->numTaps
,&self->instance->mu
,&self->instance->postShift
))
    {


    }
    return 0;
}

GETFIELD(arm_lms_instance_q15,numTaps,"h");
GETFIELD(arm_lms_instance_q15,mu,"h");
GETFIELD(arm_lms_instance_q15,postShift,"i");


static PyMethodDef arm_lms_instance_q15_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_lms_instance_q15_numTaps,METH_NOARGS,"numTaps"},
    {"mu", (PyCFunction) Method_arm_lms_instance_q15_mu,METH_NOARGS,"mu"},
    {"postShift", (PyCFunction) Method_arm_lms_instance_q15_postShift,METH_NOARGS,"postShift"},

    {NULL}  /* Sentinel */
};


DSPType(arm_lms_instance_q15,arm_lms_instance_q15_new,arm_lms_instance_q15_dealloc,arm_lms_instance_q15_init,arm_lms_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_lms_instance_q31 *instance;
} dsp_arm_lms_instance_q31Object;


static void
arm_lms_instance_q31_dealloc(dsp_arm_lms_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_lms_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_lms_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_lms_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_lms_instance_q31));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_lms_instance_q31_init(dsp_arm_lms_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps","mu","postShift",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hii", kwlist,&self->instance->numTaps
,&self->instance->mu
,&self->instance->postShift
))
    {


    }
    return 0;
}

GETFIELD(arm_lms_instance_q31,numTaps,"h");
GETFIELD(arm_lms_instance_q31,mu,"i");
GETFIELD(arm_lms_instance_q31,postShift,"i");


static PyMethodDef arm_lms_instance_q31_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_lms_instance_q31_numTaps,METH_NOARGS,"numTaps"},
    {"mu", (PyCFunction) Method_arm_lms_instance_q31_mu,METH_NOARGS,"mu"},
    {"postShift", (PyCFunction) Method_arm_lms_instance_q31_postShift,METH_NOARGS,"postShift"},

    {NULL}  /* Sentinel */
};


DSPType(arm_lms_instance_q31,arm_lms_instance_q31_new,arm_lms_instance_q31_dealloc,arm_lms_instance_q31_init,arm_lms_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_lms_norm_instance_f32 *instance;
} dsp_arm_lms_norm_instance_f32Object;


static void
arm_lms_norm_instance_f32_dealloc(dsp_arm_lms_norm_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_lms_norm_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_lms_norm_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_lms_norm_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_lms_norm_instance_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;

    }


    return (PyObject *)self;
}

static int
arm_lms_norm_instance_f32_init(dsp_arm_lms_norm_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
char *kwlist[] = {
"numTaps","mu","energy","x0",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hfff", kwlist,&self->instance->numTaps
,&self->instance->mu
,&self->instance->energy
,&self->instance->x0
))
    {


    }
    return 0;
}

GETFIELD(arm_lms_norm_instance_f32,numTaps,"h");
GETFIELD(arm_lms_norm_instance_f32,mu,"f");
GETFIELD(arm_lms_norm_instance_f32,energy,"f");
GETFIELD(arm_lms_norm_instance_f32,x0,"f");


static PyMethodDef arm_lms_norm_instance_f32_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_lms_norm_instance_f32_numTaps,METH_NOARGS,"numTaps"},
    {"mu", (PyCFunction) Method_arm_lms_norm_instance_f32_mu,METH_NOARGS,"mu"},
    {"energy", (PyCFunction) Method_arm_lms_norm_instance_f32_energy,METH_NOARGS,"energy"},
    {"x0", (PyCFunction) Method_arm_lms_norm_instance_f32_x0,METH_NOARGS,"x0"},

    {NULL}  /* Sentinel */
};


DSPType(arm_lms_norm_instance_f32,arm_lms_norm_instance_f32_new,arm_lms_norm_instance_f32_dealloc,arm_lms_norm_instance_f32_init,arm_lms_norm_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_lms_norm_instance_q31 *instance;
} dsp_arm_lms_norm_instance_q31Object;


static void
arm_lms_norm_instance_q31_dealloc(dsp_arm_lms_norm_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q31_t*)self->instance->pCoeffs);
       }



       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_lms_norm_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_lms_norm_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_lms_norm_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_lms_norm_instance_q31));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;
        self->instance->recipTable = NULL;

    }


    return (PyObject *)self;
}

static int
arm_lms_norm_instance_q31_init(dsp_arm_lms_norm_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
    PyObject *recipTable=NULL;
char *kwlist[] = {
"numTaps","mu","postShift","energy","x0",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hiiii", kwlist,&self->instance->numTaps
,&self->instance->mu
,&self->instance->postShift
,&self->instance->energy
,&self->instance->x0
))
    {


    }
    return 0;
}

GETFIELD(arm_lms_norm_instance_q31,numTaps,"h");
GETFIELD(arm_lms_norm_instance_q31,mu,"i");
GETFIELD(arm_lms_norm_instance_q31,postShift,"i");
GETFIELD(arm_lms_norm_instance_q31,energy,"i");
GETFIELD(arm_lms_norm_instance_q31,x0,"i");


static PyMethodDef arm_lms_norm_instance_q31_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_lms_norm_instance_q31_numTaps,METH_NOARGS,"numTaps"},
    {"mu", (PyCFunction) Method_arm_lms_norm_instance_q31_mu,METH_NOARGS,"mu"},
    {"postShift", (PyCFunction) Method_arm_lms_norm_instance_q31_postShift,METH_NOARGS,"postShift"},
    {"energy", (PyCFunction) Method_arm_lms_norm_instance_q31_energy,METH_NOARGS,"energy"},
    {"x0", (PyCFunction) Method_arm_lms_norm_instance_q31_x0,METH_NOARGS,"x0"},

    {NULL}  /* Sentinel */
};


DSPType(arm_lms_norm_instance_q31,arm_lms_norm_instance_q31_new,arm_lms_norm_instance_q31_dealloc,arm_lms_norm_instance_q31_init,arm_lms_norm_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_lms_norm_instance_q15 *instance;
} dsp_arm_lms_norm_instance_q15Object;


static void
arm_lms_norm_instance_q15_dealloc(dsp_arm_lms_norm_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_lms_norm_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_lms_norm_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_lms_norm_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_lms_norm_instance_q15));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;
        self->instance->recipTable = NULL;

    }


    return (PyObject *)self;
}

static int
arm_lms_norm_instance_q15_init(dsp_arm_lms_norm_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
    PyObject *recipTable=NULL;
char *kwlist[] = {
"numTaps","mu","postShift","energy","x0",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhihh", kwlist,&self->instance->numTaps
,&self->instance->mu
,&self->instance->postShift
,&self->instance->energy
,&self->instance->x0
))
    {


    }
    return 0;
}

GETFIELD(arm_lms_norm_instance_q15,numTaps,"h");
GETFIELD(arm_lms_norm_instance_q15,mu,"h");
GETFIELD(arm_lms_norm_instance_q15,postShift,"i");
GETFIELD(arm_lms_norm_instance_q15,energy,"h");
GETFIELD(arm_lms_norm_instance_q15,x0,"h");


static PyMethodDef arm_lms_norm_instance_q15_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_lms_norm_instance_q15_numTaps,METH_NOARGS,"numTaps"},
    {"mu", (PyCFunction) Method_arm_lms_norm_instance_q15_mu,METH_NOARGS,"mu"},
    {"postShift", (PyCFunction) Method_arm_lms_norm_instance_q15_postShift,METH_NOARGS,"postShift"},
    {"energy", (PyCFunction) Method_arm_lms_norm_instance_q15_energy,METH_NOARGS,"energy"},
    {"x0", (PyCFunction) Method_arm_lms_norm_instance_q15_x0,METH_NOARGS,"x0"},

    {NULL}  /* Sentinel */
};


DSPType(arm_lms_norm_instance_q15,arm_lms_norm_instance_q15_new,arm_lms_norm_instance_q15_dealloc,arm_lms_norm_instance_q15_init,arm_lms_norm_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_sparse_instance_f32 *instance;
} dsp_arm_fir_sparse_instance_f32Object;


static void
arm_fir_sparse_instance_f32_dealloc(dsp_arm_fir_sparse_instance_f32Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       if (self->instance->pTapDelay)
       {
          PyMem_Free(self->instance->pTapDelay);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_sparse_instance_f32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_sparse_instance_f32Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_sparse_instance_f32Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_sparse_instance_f32));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;
        self->instance->pTapDelay = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_sparse_instance_f32_init(dsp_arm_fir_sparse_instance_f32Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
    PyObject *pTapDelay=NULL;
char *kwlist[] = {
"numTaps","stateIndex","maxDelay","pTapDelay",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhhO", kwlist,&self->instance->numTaps
,&self->instance->stateIndex
,&self->instance->maxDelay
,&pTapDelay
))
    {

    INITARRAYFIELD(pTapDelay,NPY_INT32,int32_t,int32_t);

    }
    return 0;
}

GETFIELD(arm_fir_sparse_instance_f32,numTaps,"h");
GETFIELD(arm_fir_sparse_instance_f32,stateIndex,"h");
GETFIELD(arm_fir_sparse_instance_f32,maxDelay,"h");


static PyMethodDef arm_fir_sparse_instance_f32_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_sparse_instance_f32_numTaps,METH_NOARGS,"numTaps"},
    {"stateIndex", (PyCFunction) Method_arm_fir_sparse_instance_f32_stateIndex,METH_NOARGS,"stateIndex"},
    {"maxDelay", (PyCFunction) Method_arm_fir_sparse_instance_f32_maxDelay,METH_NOARGS,"maxDelay"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_sparse_instance_f32,arm_fir_sparse_instance_f32_new,arm_fir_sparse_instance_f32_dealloc,arm_fir_sparse_instance_f32_init,arm_fir_sparse_instance_f32_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_sparse_instance_q31 *instance;
} dsp_arm_fir_sparse_instance_q31Object;


static void
arm_fir_sparse_instance_q31_dealloc(dsp_arm_fir_sparse_instance_q31Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((float32_t*)self->instance->pCoeffs);
       }


       if (self->instance->pTapDelay)
       {
          PyMem_Free(self->instance->pTapDelay);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_sparse_instance_q31_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_sparse_instance_q31Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_sparse_instance_q31Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_sparse_instance_q31));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;
        self->instance->pTapDelay = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_sparse_instance_q31_init(dsp_arm_fir_sparse_instance_q31Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
    PyObject *pTapDelay=NULL;
char *kwlist[] = {
"numTaps","stateIndex","maxDelay","pTapDelay",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhhO", kwlist,&self->instance->numTaps
,&self->instance->stateIndex
,&self->instance->maxDelay
,&pTapDelay
))
    {

    INITARRAYFIELD(pTapDelay,NPY_INT32,int32_t,int32_t);

    }
    return 0;
}

GETFIELD(arm_fir_sparse_instance_q31,numTaps,"h");
GETFIELD(arm_fir_sparse_instance_q31,stateIndex,"h");
GETFIELD(arm_fir_sparse_instance_q31,maxDelay,"h");


static PyMethodDef arm_fir_sparse_instance_q31_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_sparse_instance_q31_numTaps,METH_NOARGS,"numTaps"},
    {"stateIndex", (PyCFunction) Method_arm_fir_sparse_instance_q31_stateIndex,METH_NOARGS,"stateIndex"},
    {"maxDelay", (PyCFunction) Method_arm_fir_sparse_instance_q31_maxDelay,METH_NOARGS,"maxDelay"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_sparse_instance_q31,arm_fir_sparse_instance_q31_new,arm_fir_sparse_instance_q31_dealloc,arm_fir_sparse_instance_q31_init,arm_fir_sparse_instance_q31_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_sparse_instance_q15 *instance;
} dsp_arm_fir_sparse_instance_q15Object;


static void
arm_fir_sparse_instance_q15_dealloc(dsp_arm_fir_sparse_instance_q15Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q15_t*)self->instance->pCoeffs);
       }


       if (self->instance->pTapDelay)
       {
          PyMem_Free(self->instance->pTapDelay);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_sparse_instance_q15_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_sparse_instance_q15Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_sparse_instance_q15Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_sparse_instance_q15));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;
        self->instance->pTapDelay = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_sparse_instance_q15_init(dsp_arm_fir_sparse_instance_q15Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
    PyObject *pTapDelay=NULL;
char *kwlist[] = {
"numTaps","stateIndex","maxDelay","pTapDelay",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhhO", kwlist,&self->instance->numTaps
,&self->instance->stateIndex
,&self->instance->maxDelay
,&pTapDelay
))
    {

    INITARRAYFIELD(pTapDelay,NPY_INT32,int32_t,int32_t);

    }
    return 0;
}

GETFIELD(arm_fir_sparse_instance_q15,numTaps,"h");
GETFIELD(arm_fir_sparse_instance_q15,stateIndex,"h");
GETFIELD(arm_fir_sparse_instance_q15,maxDelay,"h");


static PyMethodDef arm_fir_sparse_instance_q15_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_sparse_instance_q15_numTaps,METH_NOARGS,"numTaps"},
    {"stateIndex", (PyCFunction) Method_arm_fir_sparse_instance_q15_stateIndex,METH_NOARGS,"stateIndex"},
    {"maxDelay", (PyCFunction) Method_arm_fir_sparse_instance_q15_maxDelay,METH_NOARGS,"maxDelay"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_sparse_instance_q15,arm_fir_sparse_instance_q15_new,arm_fir_sparse_instance_q15_dealloc,arm_fir_sparse_instance_q15_init,arm_fir_sparse_instance_q15_methods);


typedef struct {
    PyObject_HEAD
    arm_fir_sparse_instance_q7 *instance;
} dsp_arm_fir_sparse_instance_q7Object;


static void
arm_fir_sparse_instance_q7_dealloc(dsp_arm_fir_sparse_instance_q7Object* self)
{
    //printf("Dealloc called\n");
    if (self->instance)
    {


       if (self->instance->pState)
       {
          PyMem_Free(self->instance->pState);
       }


       if (self->instance->pCoeffs)
       {
          PyMem_Free((q7_t*)self->instance->pCoeffs);
       }


       if (self->instance->pTapDelay)
       {
          PyMem_Free(self->instance->pTapDelay);
       }


       PyMem_Free(self->instance);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject *
arm_fir_sparse_instance_q7_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    dsp_arm_fir_sparse_instance_q7Object *self;
    //printf("New called\n");

    self = (dsp_arm_fir_sparse_instance_q7Object *)type->tp_alloc(type, 0);
    //printf("alloc called\n");

    if (self != NULL) {

        self->instance = PyMem_Malloc(sizeof(arm_fir_sparse_instance_q7));

        self->instance->pState = NULL;
        self->instance->pCoeffs = NULL;
        self->instance->pTapDelay = NULL;

    }


    return (PyObject *)self;
}

static int
arm_fir_sparse_instance_q7_init(dsp_arm_fir_sparse_instance_q7Object *self, PyObject *args, PyObject *kwds)
{

    PyObject *pState=NULL;
    PyObject *pCoeffs=NULL;
    PyObject *pTapDelay=NULL;
char *kwlist[] = {
"numTaps","stateIndex","maxDelay","pTapDelay",NULL
};

if (PyArg_ParseTupleAndKeywords(args, kwds, "|hhhO", kwlist,&self->instance->numTaps
,&self->instance->stateIndex
,&self->instance->maxDelay
,&pTapDelay
))
    {

    INITARRAYFIELD(pTapDelay,NPY_INT32,int32_t,int32_t);

    }
    return 0;
}

GETFIELD(arm_fir_sparse_instance_q7,numTaps,"h");
GETFIELD(arm_fir_sparse_instance_q7,stateIndex,"h");
GETFIELD(arm_fir_sparse_instance_q7,maxDelay,"h");


static PyMethodDef arm_fir_sparse_instance_q7_methods[] = {

    {"numTaps", (PyCFunction) Method_arm_fir_sparse_instance_q7_numTaps,METH_NOARGS,"numTaps"},
    {"stateIndex", (PyCFunction) Method_arm_fir_sparse_instance_q7_stateIndex,METH_NOARGS,"stateIndex"},
    {"maxDelay", (PyCFunction) Method_arm_fir_sparse_instance_q7_maxDelay,METH_NOARGS,"maxDelay"},

    {NULL}  /* Sentinel */
};


DSPType(arm_fir_sparse_instance_q7,arm_fir_sparse_instance_q7_new,arm_fir_sparse_instance_q7_dealloc,arm_fir_sparse_instance_q7_init,arm_fir_sparse_instance_q7_methods);



void typeRegistration(PyObject *module) {

  ADDTYPE(arm_fir_instance_q7);
  ADDTYPE(arm_fir_instance_q15);
  ADDTYPE(arm_fir_instance_q31);
  ADDTYPE(arm_fir_instance_f32);
  ADDTYPE(arm_fir_instance_f64);
  ADDTYPE(arm_biquad_casd_df1_inst_q15);
  ADDTYPE(arm_biquad_casd_df1_inst_q31);
  ADDTYPE(arm_biquad_casd_df1_inst_f32);
  ADDTYPE(arm_fir_decimate_instance_q15);
  ADDTYPE(arm_fir_decimate_instance_q31);
  ADDTYPE(arm_fir_decimate_instance_f32);
  ADDTYPE(arm_fir_interpolate_instance_q15);
  ADDTYPE(arm_fir_interpolate_instance_q31);
  ADDTYPE(arm_fir_interpolate_instance_f32);
  ADDTYPE(arm_biquad_cas_df1_32x64_ins_q31);
  ADDTYPE(arm_biquad_cascade_df2T_instance_f32);
  ADDTYPE(arm_biquad_cascade_stereo_df2T_instance_f32);
  ADDTYPE(arm_biquad_cascade_df2T_instance_f64);
  ADDTYPE(arm_fir_lattice_instance_q15);
  ADDTYPE(arm_fir_lattice_instance_q31);
  ADDTYPE(arm_fir_lattice_instance_f32);
  ADDTYPE(arm_iir_lattice_instance_q15);
  ADDTYPE(arm_iir_lattice_instance_q31);
  ADDTYPE(arm_iir_lattice_instance_f32);
  ADDTYPE(arm_lms_instance_f32);
  ADDTYPE(arm_lms_instance_q15);
  ADDTYPE(arm_lms_instance_q31);
  ADDTYPE(arm_lms_norm_instance_f32);
  ADDTYPE(arm_lms_norm_instance_q31);
  ADDTYPE(arm_lms_norm_instance_q15);
  ADDTYPE(arm_fir_sparse_instance_f32);
  ADDTYPE(arm_fir_sparse_instance_q31);
  ADDTYPE(arm_fir_sparse_instance_q15);
  ADDTYPE(arm_fir_sparse_instance_q7);
}







static PyObject *
cmsis_arm_fir_q7(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q7_t *pSrc_converted=NULL; // input
  q7_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_instance_q7Object *selfS = (dsp_arm_fir_instance_q7Object *)S;
    GETARGUMENT(pSrc,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q7_t)*blockSize);


    arm_fir_q7(selfS->instance,pSrc_converted,pDst,blockSize);
 INT8ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_init_q7(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q7_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q7_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_instance_q7Object *selfS = (dsp_arm_fir_instance_q7Object *)S;
    GETARGUMENT(pCoeffs,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pState,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_init_q7(selfS->instance,numTaps,pCoeffs_converted,pState_converted,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_instance_q15Object *selfS = (dsp_arm_fir_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_fir_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_instance_q15Object *selfS = (dsp_arm_fir_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_fir_fast_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_instance_q15Object *selfS = (dsp_arm_fir_instance_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_status returnValue = arm_fir_init_q15(selfS->instance,numTaps,pCoeffs_converted,pState_converted,blockSize);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_instance_q31Object *selfS = (dsp_arm_fir_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_fir_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_fast_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_instance_q31Object *selfS = (dsp_arm_fir_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_fir_fast_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_instance_q31Object *selfS = (dsp_arm_fir_instance_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_init_q31(selfS->instance,numTaps,pCoeffs_converted,pState_converted,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_instance_f32Object *selfS = (dsp_arm_fir_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_fir_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_fir_f64(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_instance_f64Object *selfS = (dsp_arm_fir_instance_f64Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_fir_f64(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_instance_f32Object *selfS = (dsp_arm_fir_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_init_f32(selfS->instance,numTaps,pCoeffs_converted,pState_converted,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}

static PyObject *
cmsis_arm_fir_init_f64(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  float64_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float64_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_instance_f64Object *selfS = (dsp_arm_fir_instance_f64Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float64_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float64_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_init_f64(selfS->instance,numTaps,pCoeffs_converted,pState_converted,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_casd_df1_inst_q15Object *selfS = (dsp_arm_biquad_casd_df1_inst_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_biquad_cascade_df1_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  int32_t postShift; // input

  if (PyArg_ParseTuple(args,"OiOOi",&S,&numStages,&pCoeffs,&pState,&postShift))
  {

    dsp_arm_biquad_casd_df1_inst_q15Object *selfS = (dsp_arm_biquad_casd_df1_inst_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);

    arm_biquad_cascade_df1_init_q15(selfS->instance,(uint8_t)numStages,pCoeffs_converted,pState_converted,(int8_t)postShift);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_casd_df1_inst_q15Object *selfS = (dsp_arm_biquad_casd_df1_inst_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_biquad_cascade_df1_fast_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_casd_df1_inst_q31Object *selfS = (dsp_arm_biquad_casd_df1_inst_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_biquad_cascade_df1_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_fast_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_casd_df1_inst_q31Object *selfS = (dsp_arm_biquad_casd_df1_inst_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_biquad_cascade_df1_fast_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  int32_t postShift; // input

  if (PyArg_ParseTuple(args,"OiOOi",&S,&numStages,&pCoeffs,&pState,&postShift))
  {

    dsp_arm_biquad_casd_df1_inst_q31Object *selfS = (dsp_arm_biquad_casd_df1_inst_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);

    arm_biquad_cascade_df1_init_q31(selfS->instance,(uint8_t)numStages,pCoeffs_converted,pState_converted,(int8_t)postShift);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_casd_df1_inst_f32Object *selfS = (dsp_arm_biquad_casd_df1_inst_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_biquad_cascade_df1_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df1_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOO",&S,&numStages,&pCoeffs,&pState))
  {

    dsp_arm_biquad_casd_df1_inst_f32Object *selfS = (dsp_arm_biquad_casd_df1_inst_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);

    arm_biquad_cascade_df1_init_f32(selfS->instance,(uint8_t)numStages,pCoeffs_converted,pState_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}



static PyObject *
cmsis_arm_levinson_durbin_q31(PyObject *obj, PyObject *args)
{

  PyObject *pPhi=NULL; // input
  q31_t *pPhi_converted=NULL; // input
  q31_t *pA=NULL; // output
  q31_t err; // output
  uint32_t nbCoefs; // input

  if (PyArg_ParseTuple(args,"Oi",&pPhi,&nbCoefs))
  {

    GETARGUMENT(pPhi,NPY_INT32,int32_t,q31_t);

    pA=PyMem_Malloc(sizeof(q31_t)*nbCoefs);


    arm_levinson_durbin_q31(pPhi_converted,pA,&err,nbCoefs);
    
    INT32ARRAY1(pAOBJ,nbCoefs,pA);

    PyObject *pythonResult = Py_BuildValue("Oi",pAOBJ,err);

    FREEARGUMENT(pPhi_converted);
    Py_DECREF(pAOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_levinson_durbin_f32(PyObject *obj, PyObject *args)
{

  PyObject *pPhi=NULL; // input
  float32_t *pPhi_converted=NULL; // input
  float32_t *pA=NULL; // output
  float32_t err; // output
  uint32_t nbCoefs; // input

  if (PyArg_ParseTuple(args,"Oi",&pPhi,&nbCoefs))
  {

    GETARGUMENT(pPhi,NPY_DOUBLE,double,float32_t);

    pA=PyMem_Malloc(sizeof(float32_t)*nbCoefs);


    arm_levinson_durbin_f32(pPhi_converted,pA,&err,nbCoefs);
    
    FLOATARRAY1(pAOBJ,nbCoefs,pA);

    PyObject *pythonResult = Py_BuildValue("Of",pAOBJ,err);

    FREEARGUMENT(pPhi_converted);
    Py_DECREF(pAOBJ);
    return(pythonResult);

  }
  return(NULL);
}



static PyObject *
cmsis_arm_conv_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  float32_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(float32_t)*outputLength);


    arm_conv_f32(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 FLOATARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_opt_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  PyObject *pScratch1=NULL; // input
  q15_t *pScratch1_converted=NULL; // input
  PyObject *pScratch2=NULL; // input
  q15_t *pScratch2_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiOO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&pScratch1,&pScratch2))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch1,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch2,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_conv_opt_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,pScratch1_converted,pScratch2_converted);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch1_converted);
    FREEARGUMENT(pScratch2_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_conv_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_conv_fast_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_fast_opt_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  PyObject *pScratch1=NULL; // input
  q15_t *pScratch1_converted=NULL; // input
  PyObject *pScratch2=NULL; // input
  q15_t *pScratch2_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiOO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&pScratch1,&pScratch2))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch1,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch2,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_conv_fast_opt_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,pScratch1_converted,pScratch2_converted);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch1_converted);
    FREEARGUMENT(pScratch2_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q31_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q31_t)*outputLength);


    arm_conv_q31(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT32ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_fast_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q31_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q31_t)*outputLength);


    arm_conv_fast_q31(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT32ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_opt_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q7_t *pDst=NULL; // output
  PyObject *pScratch1=NULL; // input
  q15_t *pScratch1_converted=NULL; // input
  PyObject *pScratch2=NULL; // input
  q15_t *pScratch2_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiOO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&pScratch1,&pScratch2))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pScratch1,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch2,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q7_t)*outputLength);


    arm_conv_opt_q7(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,pScratch1_converted,pScratch2_converted);
 INT8ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch1_converted);
    FREEARGUMENT(pScratch2_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q7_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q7_t)*outputLength);


    arm_conv_q7(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT8ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_partial_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  float32_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input

  if (PyArg_ParseTuple(args,"OiOiii",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(float32_t)*outputLength);


    arm_status returnValue = arm_conv_partial_f32(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 FLOATARRAY1(pDstOBJ,outputLength,pDst);

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
cmsis_arm_conv_partial_opt_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input
  PyObject *pScratch1=NULL; // input
  q15_t *pScratch1_converted=NULL; // input
  PyObject *pScratch2=NULL; // input
  q15_t *pScratch2_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiiiOO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints,&pScratch1,&pScratch2))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch1,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch2,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_status returnValue = arm_conv_partial_opt_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints,pScratch1_converted,pScratch2_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch1_converted);
    FREEARGUMENT(pScratch2_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_partial_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input

  if (PyArg_ParseTuple(args,"OiOiii",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_status returnValue = arm_conv_partial_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

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
cmsis_arm_conv_partial_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input

  if (PyArg_ParseTuple(args,"OiOiii",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_status returnValue = arm_conv_partial_fast_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

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
cmsis_arm_conv_partial_fast_opt_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input
  PyObject *pScratch1=NULL; // input
  q15_t *pScratch1_converted=NULL; // input
  PyObject *pScratch2=NULL; // input
  q15_t *pScratch2_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiiiOO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints,&pScratch1,&pScratch2))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch1,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch2,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_status returnValue = arm_conv_partial_fast_opt_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints,pScratch1_converted,pScratch2_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch1_converted);
    FREEARGUMENT(pScratch2_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_partial_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q31_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input

  if (PyArg_ParseTuple(args,"OiOiii",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q31_t)*outputLength);


    arm_status returnValue = arm_conv_partial_q31(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT32ARRAY1(pDstOBJ,outputLength,pDst);

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
cmsis_arm_conv_partial_fast_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q31_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input

  if (PyArg_ParseTuple(args,"OiOiii",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q31_t)*outputLength);


    arm_status returnValue = arm_conv_partial_fast_q31(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT32ARRAY1(pDstOBJ,outputLength,pDst);

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
cmsis_arm_conv_partial_opt_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q7_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input
  PyObject *pScratch1=NULL; // input
  q15_t *pScratch1_converted=NULL; // input
  PyObject *pScratch2=NULL; // input
  q15_t *pScratch2_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiiiOO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints,&pScratch1,&pScratch2))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pScratch1,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch2,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q7_t)*outputLength);


    arm_status returnValue = arm_conv_partial_opt_q7(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints,pScratch1_converted,pScratch2_converted);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT8ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("OO",theReturnOBJ,pDstOBJ);

    Py_DECREF(theReturnOBJ);
    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch1_converted);
    FREEARGUMENT(pScratch2_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_conv_partial_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q7_t *pDst=NULL; // output
  uint32_t firstIndex; // input
  uint32_t numPoints; // input

  if (PyArg_ParseTuple(args,"OiOiii",&pSrcA,&srcALen,&pSrcB,&srcBLen,&firstIndex,&numPoints))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    uint32_t outputLength = srcALen + srcBLen - 1 ;

    pDst=PyMem_Malloc(sizeof(q7_t)*outputLength);


    arm_status returnValue = arm_conv_partial_q7(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,firstIndex,numPoints);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);
 INT8ARRAY1(pDstOBJ,outputLength,pDst);

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
cmsis_arm_fir_decimate_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_decimate_instance_f32Object *selfS = (dsp_arm_fir_decimate_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_fir_decimate_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_decimate_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  uint32_t M; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhiOO",&S,&numTaps,&M,&pCoeffs,&pState))
  {

    dsp_arm_fir_decimate_instance_f32Object *selfS = (dsp_arm_fir_decimate_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_status returnValue = arm_fir_decimate_init_f32(selfS->instance,numTaps,(uint8_t)M,pCoeffs_converted,pState_converted,blockSize);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_decimate_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_decimate_instance_q15Object *selfS = (dsp_arm_fir_decimate_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_fir_decimate_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_decimate_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_decimate_instance_q15Object *selfS = (dsp_arm_fir_decimate_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_fir_decimate_fast_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_decimate_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  uint32_t M; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhiOO",&S,&numTaps,&M,&pCoeffs,&pState))
  {

    dsp_arm_fir_decimate_instance_q15Object *selfS = (dsp_arm_fir_decimate_instance_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_status returnValue = arm_fir_decimate_init_q15(selfS->instance,numTaps,(uint8_t)M,pCoeffs_converted,pState_converted,blockSize);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_decimate_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_decimate_instance_q31Object *selfS = (dsp_arm_fir_decimate_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_fir_decimate_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_decimate_fast_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_decimate_instance_q31Object *selfS = (dsp_arm_fir_decimate_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_fir_decimate_fast_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_decimate_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  uint32_t M; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhiOO",&S,&numTaps,&M,&pCoeffs,&pState))
  {

    dsp_arm_fir_decimate_instance_q31Object *selfS = (dsp_arm_fir_decimate_instance_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_status returnValue = arm_fir_decimate_init_q31(selfS->instance,numTaps,(uint8_t)M,pCoeffs_converted,pState_converted,blockSize);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_interpolate_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_interpolate_instance_q15Object *selfS = (dsp_arm_fir_interpolate_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_fir_interpolate_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_interpolate_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t L; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OihOO",&S,&L,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_interpolate_instance_q15Object *selfS = (dsp_arm_fir_interpolate_instance_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_status returnValue = arm_fir_interpolate_init_q15(selfS->instance,(uint8_t)L,numTaps,pCoeffs_converted,pState_converted,blockSize);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_interpolate_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_interpolate_instance_q31Object *selfS = (dsp_arm_fir_interpolate_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_fir_interpolate_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_interpolate_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t L; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OihOO",&S,&L,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_interpolate_instance_q31Object *selfS = (dsp_arm_fir_interpolate_instance_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_status returnValue = arm_fir_interpolate_init_q31(selfS->instance,(uint8_t)L,numTaps,pCoeffs_converted,pState_converted,blockSize);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_interpolate_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_interpolate_instance_f32Object *selfS = (dsp_arm_fir_interpolate_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_fir_interpolate_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_interpolate_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t L; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OihOO",&S,&L,&numTaps,&pCoeffs,&pState))
  {

    dsp_arm_fir_interpolate_instance_f32Object *selfS = (dsp_arm_fir_interpolate_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_status returnValue = arm_fir_interpolate_init_f32(selfS->instance,(uint8_t)L,numTaps,pCoeffs_converted,pState_converted,blockSize);
    PyObject* theReturnOBJ=Py_BuildValue("i",returnValue);

    PyObject *pythonResult = Py_BuildValue("O",theReturnOBJ);

    Py_DECREF(theReturnOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cas_df1_32x64_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  arm_biquad_cas_df1_32x64_ins_q31 *S_converted=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_biquad_cas_df1_32x64_q31(S_converted,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cas_df1_32x64_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  arm_biquad_cas_df1_32x64_ins_q31 *S_converted=NULL; // input
  uint32_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q63_t *pState_converted=NULL; // input
  uint32_t postShift; // input

  if (PyArg_ParseTuple(args,"OiOOi",&S,&numStages,&pCoeffs,&pState,&postShift))
  {

    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT64,q63_t,q63_t);

    arm_biquad_cas_df1_32x64_init_q31(S_converted,(uint8_t)numStages,pCoeffs_converted,pState_converted,(uint8_t)postShift);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df2T_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_cascade_df2T_instance_f32Object *selfS = (dsp_arm_biquad_cascade_df2T_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_biquad_cascade_df2T_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_stereo_df2T_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object *selfS = (dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_biquad_cascade_stereo_df2T_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df2T_f64(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float64_t *pSrc_converted=NULL; // input
  float64_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_biquad_cascade_df2T_instance_f64Object *selfS = (dsp_arm_biquad_cascade_df2T_instance_f64Object *)S;
    GETARGUMENT(pSrc,NPY_FLOAT64,float64_t,float64_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float64_t)*blockSize);


    arm_biquad_cascade_df2T_f64(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOAT64ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df2T_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOO",&S,&numStages,&pCoeffs,&pState))
  {

    dsp_arm_biquad_cascade_df2T_instance_f32Object *selfS = (dsp_arm_biquad_cascade_df2T_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);

    arm_biquad_cascade_df2T_init_f32(selfS->instance,(uint8_t)numStages,pCoeffs_converted,pState_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_stereo_df2T_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOO",&S,&numStages,&pCoeffs,&pState))
  {

    dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object *selfS = (dsp_arm_biquad_cascade_stereo_df2T_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);

    arm_biquad_cascade_stereo_df2T_init_f32(selfS->instance,(uint8_t)numStages,pCoeffs_converted,pState_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_biquad_cascade_df2T_init_f64(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint32_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  float64_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float64_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOO",&S,&numStages,&pCoeffs,&pState))
  {

    dsp_arm_biquad_cascade_df2T_instance_f64Object *selfS = (dsp_arm_biquad_cascade_df2T_instance_f64Object *)S;
    GETARGUMENT(pCoeffs,NPY_FLOAT64,float64_t,float64_t);
    GETARGUMENT(pState,NPY_FLOAT64,float64_t,float64_t);

    arm_biquad_cascade_df2T_init_f64(selfS->instance,(uint8_t)numStages,pCoeffs_converted,pState_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_lattice_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numStages,&pCoeffs,&pState))
  {

    dsp_arm_fir_lattice_instance_q15Object *selfS = (dsp_arm_fir_lattice_instance_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);

    arm_fir_lattice_init_q15(selfS->instance,numStages,pCoeffs_converted,pState_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_lattice_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_lattice_instance_q15Object *selfS = (dsp_arm_fir_lattice_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_fir_lattice_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_lattice_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numStages,&pCoeffs,&pState))
  {

    dsp_arm_fir_lattice_instance_q31Object *selfS = (dsp_arm_fir_lattice_instance_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);

    arm_fir_lattice_init_q31(selfS->instance,numStages,pCoeffs_converted,pState_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_lattice_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_lattice_instance_q31Object *selfS = (dsp_arm_fir_lattice_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_fir_lattice_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_lattice_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numStages; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OhOO",&S,&numStages,&pCoeffs,&pState))
  {

    dsp_arm_fir_lattice_instance_f32Object *selfS = (dsp_arm_fir_lattice_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);

    arm_fir_lattice_init_f32(selfS->instance,numStages,pCoeffs_converted,pState_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_lattice_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_fir_lattice_instance_f32Object *selfS = (dsp_arm_fir_lattice_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_fir_lattice_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_iir_lattice_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_iir_lattice_instance_f32Object *selfS = (dsp_arm_iir_lattice_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_iir_lattice_f32(selfS->instance,pSrc_converted,pDst,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_iir_lattice_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numStages; // input
  PyObject *pkCoeffs=NULL; // input
  float32_t *pkCoeffs_converted=NULL; // input
  PyObject *pvCoeffs=NULL; // input
  float32_t *pvCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOO",&S,&numStages,&pkCoeffs,&pvCoeffs,&pState))
  {

    dsp_arm_iir_lattice_instance_f32Object *selfS = (dsp_arm_iir_lattice_instance_f32Object *)S;
    GETARGUMENT(pkCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pvCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepkCoeffs ;

    arm_iir_lattice_init_f32(selfS->instance,numStages,pkCoeffs_converted,pvCoeffs_converted,pState_converted,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_iir_lattice_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  q31_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_iir_lattice_instance_q31Object *selfS = (dsp_arm_iir_lattice_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_iir_lattice_q31(selfS->instance,pSrc_converted,pDst,blockSize);
 INT32ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_iir_lattice_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numStages; // input
  PyObject *pkCoeffs=NULL; // input
  q31_t *pkCoeffs_converted=NULL; // input
  PyObject *pvCoeffs=NULL; // input
  q31_t *pvCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOO",&S,&numStages,&pkCoeffs,&pvCoeffs,&pState))
  {

    dsp_arm_iir_lattice_instance_q31Object *selfS = (dsp_arm_iir_lattice_instance_q31Object *)S;
    GETARGUMENT(pkCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pvCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepkCoeffs ;

    arm_iir_lattice_init_q31(selfS->instance,numStages,pkCoeffs_converted,pvCoeffs_converted,pState_converted,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_iir_lattice_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  q15_t *pDst=NULL; // output
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OO",&S,&pSrc))
  {

    dsp_arm_iir_lattice_instance_q15Object *selfS = (dsp_arm_iir_lattice_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_iir_lattice_q15(selfS->instance,pSrc_converted,pDst,blockSize);
 INT16ARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_iir_lattice_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numStages; // input
  PyObject *pkCoeffs=NULL; // input
  q15_t *pkCoeffs_converted=NULL; // input
  PyObject *pvCoeffs=NULL; // input
  q15_t *pvCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOO",&S,&numStages,&pkCoeffs,&pvCoeffs,&pState))
  {

    dsp_arm_iir_lattice_instance_q15Object *selfS = (dsp_arm_iir_lattice_instance_q15Object *)S;
    GETARGUMENT(pkCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pvCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepkCoeffs ;

    arm_iir_lattice_init_q15(selfS->instance,numStages,pkCoeffs_converted,pvCoeffs_converted,pState_converted,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}



static PyObject *
cmsis_arm_lms_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  PyObject *pRef=NULL; // input
  float32_t *pRef_converted=NULL; // input
  float32_t *pOut=NULL; // output
  PyObject *pErr=NULL; // input
  float32_t *pErr_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OOOO",&S,&pSrc,&pRef,&pErr))
  {

    dsp_arm_lms_instance_f32Object *selfS = (dsp_arm_lms_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pRef,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pErr,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pOut=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_lms_f32(selfS->instance,pSrc_converted,pRef_converted,pOut,pErr_converted,blockSize);
 FLOATARRAY1(pOutOBJ,blockSize,pOut);

    PyObject *pythonResult = Py_BuildValue("O",pOutOBJ);

    FREEARGUMENT(pSrc_converted);
    FREEARGUMENT(pRef_converted);
    Py_DECREF(pOutOBJ);
    FREEARGUMENT(pErr_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input
  float32_t mu; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOf",&S,&numTaps,&pCoeffs,&pState,&mu))
  {

    dsp_arm_lms_instance_f32Object *selfS = (dsp_arm_lms_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1 ;

    arm_lms_init_f32(selfS->instance,numTaps,pCoeffs_converted,pState_converted,mu,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  q15_t mu; // input
  uint32_t blockSize; // input
  uint32_t postShift; // input

  if (PyArg_ParseTuple(args,"OhOOhi",&S,&numTaps,&pCoeffs,&pState,&mu,&postShift))
  {

    dsp_arm_lms_instance_q15Object *selfS = (dsp_arm_lms_instance_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1 ;

    arm_lms_init_q15(selfS->instance,numTaps,pCoeffs_converted,pState_converted,mu,blockSize,postShift);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  PyObject *pRef=NULL; // input
  q15_t *pRef_converted=NULL; // input
  q15_t *pOut=NULL; // output
  PyObject *pErr=NULL; // input
  q15_t *pErr_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OOOO",&S,&pSrc,&pRef,&pErr))
  {

    dsp_arm_lms_instance_q15Object *selfS = (dsp_arm_lms_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pRef,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pErr,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pOut=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_lms_q15(selfS->instance,pSrc_converted,pRef_converted,pOut,pErr_converted,blockSize);
 INT16ARRAY1(pOutOBJ,blockSize,pOut);

    PyObject *pythonResult = Py_BuildValue("O",pOutOBJ);

    FREEARGUMENT(pSrc_converted);
    FREEARGUMENT(pRef_converted);
    Py_DECREF(pOutOBJ);
    FREEARGUMENT(pErr_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  PyObject *pRef=NULL; // input
  q31_t *pRef_converted=NULL; // input
  q31_t *pOut=NULL; // output
  PyObject *pErr=NULL; // input
  q31_t *pErr_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OOOO",&S,&pSrc,&pRef,&pErr))
  {

    dsp_arm_lms_instance_q31Object *selfS = (dsp_arm_lms_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pRef,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pErr,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pOut=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_lms_q31(selfS->instance,pSrc_converted,pRef_converted,pOut,pErr_converted,blockSize);
 INT32ARRAY1(pOutOBJ,blockSize,pOut);

    PyObject *pythonResult = Py_BuildValue("O",pOutOBJ);

    FREEARGUMENT(pSrc_converted);
    FREEARGUMENT(pRef_converted);
    Py_DECREF(pOutOBJ);
    FREEARGUMENT(pErr_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  q31_t mu; // input
  uint32_t blockSize; // input
  uint32_t postShift; // input

  if (PyArg_ParseTuple(args,"OhOOii",&S,&numTaps,&pCoeffs,&pState,&mu,&postShift))
  {

    dsp_arm_lms_instance_q31Object *selfS = (dsp_arm_lms_instance_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1 ;

    arm_lms_init_q31(selfS->instance,numTaps,pCoeffs_converted,pState_converted,mu,blockSize,postShift);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_norm_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  PyObject *pRef=NULL; // input
  float32_t *pRef_converted=NULL; // input
  float32_t *pOut=NULL; // output
  PyObject *pErr=NULL; // input
  float32_t *pErr_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OOOO",&S,&pSrc,&pRef,&pErr))
  {

    dsp_arm_lms_norm_instance_f32Object *selfS = (dsp_arm_lms_norm_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pRef,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pErr,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pOut=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_lms_norm_f32(selfS->instance,pSrc_converted,pRef_converted,pOut,pErr_converted,blockSize);
 FLOATARRAY1(pOutOBJ,blockSize,pOut);

    PyObject *pythonResult = Py_BuildValue("O",pOutOBJ);

    FREEARGUMENT(pSrc_converted);
    FREEARGUMENT(pRef_converted);
    Py_DECREF(pOutOBJ);
    FREEARGUMENT(pErr_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_norm_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input
  float32_t mu; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOf",&S,&numTaps,&pCoeffs,&pState,&mu))
  {

    dsp_arm_lms_norm_instance_f32Object *selfS = (dsp_arm_lms_norm_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1 ;

    arm_lms_norm_init_f32(selfS->instance,numTaps,pCoeffs_converted,pState_converted,mu,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_norm_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q31_t *pSrc_converted=NULL; // input
  PyObject *pRef=NULL; // input
  q31_t *pRef_converted=NULL; // input
  q31_t *pOut=NULL; // output
  PyObject *pErr=NULL; // input
  q31_t *pErr_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OOOO",&S,&pSrc,&pRef,&pErr))
  {

    dsp_arm_lms_norm_instance_q31Object *selfS = (dsp_arm_lms_norm_instance_q31Object *)S;
    GETARGUMENT(pSrc,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pRef,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pErr,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepSrc ;

    pOut=PyMem_Malloc(sizeof(q31_t)*blockSize);


    arm_lms_norm_q31(selfS->instance,pSrc_converted,pRef_converted,pOut,pErr_converted,blockSize);
 INT32ARRAY1(pOutOBJ,blockSize,pOut);

    PyObject *pythonResult = Py_BuildValue("O",pOutOBJ);

    FREEARGUMENT(pSrc_converted);
    FREEARGUMENT(pRef_converted);
    Py_DECREF(pOutOBJ);
    FREEARGUMENT(pErr_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_norm_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  q31_t mu; // input
  uint32_t blockSize; // input
  uint32_t postShift; // input

  if (PyArg_ParseTuple(args,"OhOOii",&S,&numTaps,&pCoeffs,&pState,&mu,&postShift))
  {

    dsp_arm_lms_norm_instance_q31Object *selfS = (dsp_arm_lms_norm_instance_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1 ;

    arm_lms_norm_init_q31(selfS->instance,numTaps,pCoeffs_converted,pState_converted,mu,blockSize,(uint8_t)postShift);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_norm_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  q15_t *pSrc_converted=NULL; // input
  PyObject *pRef=NULL; // input
  q15_t *pRef_converted=NULL; // input
  q15_t *pOut=NULL; // output
  PyObject *pErr=NULL; // input
  q15_t *pErr_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OOOO",&S,&pSrc,&pRef,&pErr))
  {

    dsp_arm_lms_norm_instance_q15Object *selfS = (dsp_arm_lms_norm_instance_q15Object *)S;
    GETARGUMENT(pSrc,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pRef,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pErr,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepSrc ;

    pOut=PyMem_Malloc(sizeof(q15_t)*blockSize);


    arm_lms_norm_q15(selfS->instance,pSrc_converted,pRef_converted,pOut,pErr_converted,blockSize);
 INT16ARRAY1(pOutOBJ,blockSize,pOut);

    PyObject *pythonResult = Py_BuildValue("O",pOutOBJ);

    FREEARGUMENT(pSrc_converted);
    FREEARGUMENT(pRef_converted);
    Py_DECREF(pOutOBJ);
    FREEARGUMENT(pErr_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_lms_norm_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  q15_t mu; // input
  uint32_t blockSize; // input
  uint32_t postShift; // input

  if (PyArg_ParseTuple(args,"OhOOhi",&S,&numTaps,&pCoeffs,&pState,&mu,&postShift))
  {

    dsp_arm_lms_norm_instance_q15Object *selfS = (dsp_arm_lms_norm_instance_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1 ;

    arm_lms_norm_init_q15(selfS->instance,numTaps,pCoeffs_converted,pState_converted,mu,blockSize,(uint8_t)postShift);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_f32(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float32_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  float32_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  float32_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float32_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(float32_t)*outputLength);


    arm_correlate_f32(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 FLOATARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_correlate_f64(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  float64_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  float64_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  float64_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_DOUBLE,double,float64_t);
    GETARGUMENT(pSrcB,NPY_DOUBLE,double,float64_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(float64_t)*outputLength);


    arm_correlate_f64(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 FLOAT64ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}

static PyObject *
cmsis_arm_correlate_opt_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  PyObject *pScratch=NULL; // input
  q15_t *pScratch_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&pScratch))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_correlate_opt_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,pScratch_converted);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_correlate_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_fast_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_correlate_fast_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_fast_opt_q15(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q15_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q15_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q15_t *pDst=NULL; // output
  PyObject *pScratch=NULL; // input
  q15_t *pScratch_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&pScratch))
  {

    GETARGUMENT(pSrcA,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pSrcB,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q15_t)*outputLength);


    arm_correlate_fast_opt_q15(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,pScratch_converted);
 INT16ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q31_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q31_t)*outputLength);


    arm_correlate_q31(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT32ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_fast_q31(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q31_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q31_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q31_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pSrcB,NPY_INT32,int32_t,int32_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q31_t)*outputLength);


    arm_correlate_fast_q31(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT32ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_opt_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q7_t *pDst=NULL; // output
  PyObject *pScratch1=NULL; // input
  q15_t *pScratch1_converted=NULL; // input
  PyObject *pScratch2=NULL; // input
  q15_t *pScratch2_converted=NULL; // input

  if (PyArg_ParseTuple(args,"OiOiOO",&pSrcA,&srcALen,&pSrcB,&srcBLen,&pScratch1,&pScratch2))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pScratch1,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pScratch2,NPY_INT16,int16_t,int16_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q7_t)*outputLength);


    arm_correlate_opt_q7(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst,pScratch1_converted,pScratch2_converted);
 INT8ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratch1_converted);
    FREEARGUMENT(pScratch2_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_correlate_q7(PyObject *obj, PyObject *args)
{

  PyObject *pSrcA=NULL; // input
  q7_t *pSrcA_converted=NULL; // input
  uint32_t srcALen; // input
  PyObject *pSrcB=NULL; // input
  q7_t *pSrcB_converted=NULL; // input
  uint32_t srcBLen; // input
  q7_t *pDst=NULL; // output

  if (PyArg_ParseTuple(args,"OiOi",&pSrcA,&srcALen,&pSrcB,&srcBLen))
  {

    GETARGUMENT(pSrcA,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pSrcB,NPY_BYTE,int8_t,q7_t);
    uint32_t outputLength = 2*MAX(srcALen,srcBLen) - 1 ;

    pDst=PyMem_Malloc(sizeof(q7_t)*outputLength);


    arm_correlate_q7(pSrcA_converted,srcALen,pSrcB_converted,srcBLen,pDst);
 INT8ARRAY1(pDstOBJ,outputLength,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrcA_converted);
    FREEARGUMENT(pSrcB_converted);
    Py_DECREF(pDstOBJ);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_sparse_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  PyObject *pSrc=NULL; // input
  float32_t *pSrc_converted=NULL; // input
  float32_t *pDst=NULL; // output
  PyObject *pScratchIn=NULL; // input
  float32_t *pScratchIn_converted=NULL; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OOO",&S,&pSrc,&pScratchIn))
  {

    dsp_arm_fir_sparse_instance_f32Object *selfS = (dsp_arm_fir_sparse_instance_f32Object *)S;
    GETARGUMENT(pSrc,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pScratchIn,NPY_DOUBLE,double,float32_t);
    blockSize = arraySizepSrc ;

    pDst=PyMem_Malloc(sizeof(float32_t)*blockSize);


    arm_fir_sparse_f32(selfS->instance,pSrc_converted,pDst,pScratchIn_converted,blockSize);
 FLOATARRAY1(pDstOBJ,blockSize,pDst);

    PyObject *pythonResult = Py_BuildValue("O",pDstOBJ);

    FREEARGUMENT(pSrc_converted);
    Py_DECREF(pDstOBJ);
    FREEARGUMENT(pScratchIn_converted);
    return(pythonResult);

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_sparse_init_f32(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  float32_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  float32_t *pState_converted=NULL; // input
  PyObject *pTapDelay=NULL; // input
  int32_t *pTapDelay_converted=NULL; // input
  uint16_t maxDelay; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOOh",&S,&numTaps,&pCoeffs,&pState,&pTapDelay,&maxDelay))
  {

    dsp_arm_fir_sparse_instance_f32Object *selfS = (dsp_arm_fir_sparse_instance_f32Object *)S;
    GETARGUMENT(pCoeffs,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pState,NPY_DOUBLE,double,float32_t);
    GETARGUMENT(pTapDelay,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_sparse_init_f32(selfS->instance,numTaps,pCoeffs_converted,pState_converted,pTapDelay_converted,maxDelay,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_sparse_init_q31(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q31_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q31_t *pState_converted=NULL; // input
  PyObject *pTapDelay=NULL; // input
  int32_t *pTapDelay_converted=NULL; // input
  uint16_t maxDelay; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOOh",&S,&numTaps,&pCoeffs,&pState,&pTapDelay,&maxDelay))
  {

    dsp_arm_fir_sparse_instance_q31Object *selfS = (dsp_arm_fir_sparse_instance_q31Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pState,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(pTapDelay,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_sparse_init_q31(selfS->instance,numTaps,pCoeffs_converted,pState_converted,pTapDelay_converted,maxDelay,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_sparse_init_q15(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q15_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q15_t *pState_converted=NULL; // input
  PyObject *pTapDelay=NULL; // input
  int32_t *pTapDelay_converted=NULL; // input
  uint16_t maxDelay; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOOh",&S,&numTaps,&pCoeffs,&pState,&pTapDelay,&maxDelay))
  {

    dsp_arm_fir_sparse_instance_q15Object *selfS = (dsp_arm_fir_sparse_instance_q15Object *)S;
    GETARGUMENT(pCoeffs,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pState,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(pTapDelay,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_sparse_init_q15(selfS->instance,numTaps,pCoeffs_converted,pState_converted,pTapDelay_converted,maxDelay,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_fir_sparse_init_q7(PyObject *obj, PyObject *args)
{

  PyObject *S=NULL; // input
  uint16_t numTaps; // input
  PyObject *pCoeffs=NULL; // input
  q7_t *pCoeffs_converted=NULL; // input
  PyObject *pState=NULL; // input
  q7_t *pState_converted=NULL; // input
  PyObject *pTapDelay=NULL; // input
  int32_t *pTapDelay_converted=NULL; // input
  uint16_t maxDelay; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OhOOOh",&S,&numTaps,&pCoeffs,&pState,&pTapDelay,&maxDelay))
  {

    dsp_arm_fir_sparse_instance_q7Object *selfS = (dsp_arm_fir_sparse_instance_q7Object *)S;
    GETARGUMENT(pCoeffs,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pState,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(pTapDelay,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizepState - arraySizepCoeffs + 1;

    arm_fir_sparse_init_q7(selfS->instance,numTaps,pCoeffs_converted,pState_converted,pTapDelay_converted,maxDelay,blockSize);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_circularWrite_f32(PyObject *obj, PyObject *args)
{

  PyObject *circBuffer=NULL; // input
  int32_t *circBuffer_converted=NULL; // input
  int32_t L; // input
  PyObject *writeOffset=NULL; // input
  uint16_t *writeOffset_converted=NULL; // input
  int32_t bufferInc; // input
  PyObject *src=NULL; // input
  int32_t *src_converted=NULL; // input
  int32_t srcInc; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OiOiOi",&circBuffer,&L,&writeOffset,&bufferInc,&src,&srcInc))
  {

    GETARGUMENT(circBuffer,NPY_INT32,int32_t,int32_t);
    GETARGUMENT(writeOffset,NPY_UINT16,uint16_t,uint16_t);
    GETARGUMENT(src,NPY_INT32,int32_t,int32_t);
    blockSize = arraySizecircBuffer ;

    arm_circularWrite_f32(circBuffer_converted,L,writeOffset_converted,bufferInc,src_converted,srcInc,blockSize);
    FREEARGUMENT(circBuffer_converted);
    FREEARGUMENT(writeOffset_converted);
    FREEARGUMENT(src_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_circularWrite_q15(PyObject *obj, PyObject *args)
{

  PyObject *circBuffer=NULL; // input
  q15_t *circBuffer_converted=NULL; // input
  int32_t L; // input
  PyObject *writeOffset=NULL; // input
  uint16_t *writeOffset_converted=NULL; // input
  int32_t bufferInc; // input
  PyObject *src=NULL; // input
  q15_t *src_converted=NULL; // input
  int32_t srcInc; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OiOiOi",&circBuffer,&L,&writeOffset,&bufferInc,&src,&srcInc))
  {

    GETARGUMENT(circBuffer,NPY_INT16,int16_t,int16_t);
    GETARGUMENT(writeOffset,NPY_UINT16,uint16_t,uint16_t);
    GETARGUMENT(src,NPY_INT16,int16_t,int16_t);
    blockSize = arraySizecircBuffer ;

    arm_circularWrite_q15(circBuffer_converted,L,writeOffset_converted,bufferInc,src_converted,srcInc,blockSize);
    FREEARGUMENT(circBuffer_converted);
    FREEARGUMENT(writeOffset_converted);
    FREEARGUMENT(src_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}


static PyObject *
cmsis_arm_circularWrite_q7(PyObject *obj, PyObject *args)
{

  PyObject *circBuffer=NULL; // input
  q7_t *circBuffer_converted=NULL; // input
  int32_t L; // input
  PyObject *writeOffset=NULL; // input
  uint16_t *writeOffset_converted=NULL; // input
  int32_t bufferInc; // input
  PyObject *src=NULL; // input
  q7_t *src_converted=NULL; // input
  int32_t srcInc; // input
  uint32_t blockSize; // input

  if (PyArg_ParseTuple(args,"OiOiOi",&circBuffer,&L,&writeOffset,&bufferInc,&src,&srcInc))
  {

    GETARGUMENT(circBuffer,NPY_BYTE,int8_t,q7_t);
    GETARGUMENT(writeOffset,NPY_UINT16,uint16_t,uint16_t);
    GETARGUMENT(src,NPY_BYTE,int8_t,q7_t);
    blockSize = arraySizecircBuffer ;

    arm_circularWrite_q7(circBuffer_converted,L,writeOffset_converted,bufferInc,src_converted,srcInc,blockSize);
    FREEARGUMENT(circBuffer_converted);
    FREEARGUMENT(writeOffset_converted);
    FREEARGUMENT(src_converted);
    Py_RETURN_NONE;

  }
  return(NULL);
}

static PyMethodDef CMSISDSPMethods[] = {


{"arm_fir_q7",  cmsis_arm_fir_q7, METH_VARARGS,""},
{"arm_fir_init_q7",  cmsis_arm_fir_init_q7, METH_VARARGS,""},
{"arm_fir_q15",  cmsis_arm_fir_q15, METH_VARARGS,""},
{"arm_fir_fast_q15",  cmsis_arm_fir_fast_q15, METH_VARARGS,""},
{"arm_fir_init_q15",  cmsis_arm_fir_init_q15, METH_VARARGS,""},
{"arm_fir_q31",  cmsis_arm_fir_q31, METH_VARARGS,""},
{"arm_fir_fast_q31",  cmsis_arm_fir_fast_q31, METH_VARARGS,""},
{"arm_fir_init_q31",  cmsis_arm_fir_init_q31, METH_VARARGS,""},
{"arm_fir_f32",  cmsis_arm_fir_f32, METH_VARARGS,""},
{"arm_fir_init_f32",  cmsis_arm_fir_init_f32, METH_VARARGS,""},
{"arm_fir_f64",  cmsis_arm_fir_f64, METH_VARARGS,""},
{"arm_fir_init_f64",  cmsis_arm_fir_init_f64, METH_VARARGS,""},
{"arm_biquad_cascade_df1_q15",  cmsis_arm_biquad_cascade_df1_q15, METH_VARARGS,""},
{"arm_biquad_cascade_df1_init_q15",  cmsis_arm_biquad_cascade_df1_init_q15, METH_VARARGS,""},
{"arm_biquad_cascade_df1_fast_q15",  cmsis_arm_biquad_cascade_df1_fast_q15, METH_VARARGS,""},
{"arm_biquad_cascade_df1_q31",  cmsis_arm_biquad_cascade_df1_q31, METH_VARARGS,""},
{"arm_biquad_cascade_df1_fast_q31",  cmsis_arm_biquad_cascade_df1_fast_q31, METH_VARARGS,""},
{"arm_biquad_cascade_df1_init_q31",  cmsis_arm_biquad_cascade_df1_init_q31, METH_VARARGS,""},
{"arm_biquad_cascade_df1_f32",  cmsis_arm_biquad_cascade_df1_f32, METH_VARARGS,""},
{"arm_biquad_cascade_df1_init_f32",  cmsis_arm_biquad_cascade_df1_init_f32, METH_VARARGS,""},



{"arm_conv_f32",  cmsis_arm_conv_f32, METH_VARARGS,""},
{"arm_conv_opt_q15",  cmsis_arm_conv_opt_q15, METH_VARARGS,""},
{"arm_conv_q15",  cmsis_arm_conv_q15, METH_VARARGS,""},
{"arm_conv_fast_q15",  cmsis_arm_conv_fast_q15, METH_VARARGS,""},
{"arm_conv_fast_opt_q15",  cmsis_arm_conv_fast_opt_q15, METH_VARARGS,""},
{"arm_conv_q31",  cmsis_arm_conv_q31, METH_VARARGS,""},
{"arm_conv_fast_q31",  cmsis_arm_conv_fast_q31, METH_VARARGS,""},
{"arm_conv_opt_q7",  cmsis_arm_conv_opt_q7, METH_VARARGS,""},
{"arm_conv_q7",  cmsis_arm_conv_q7, METH_VARARGS,""},
{"arm_conv_partial_f32",  cmsis_arm_conv_partial_f32, METH_VARARGS,""},
{"arm_conv_partial_opt_q15",  cmsis_arm_conv_partial_opt_q15, METH_VARARGS,""},
{"arm_conv_partial_q15",  cmsis_arm_conv_partial_q15, METH_VARARGS,""},
{"arm_conv_partial_fast_q15",  cmsis_arm_conv_partial_fast_q15, METH_VARARGS,""},
{"arm_conv_partial_fast_opt_q15",  cmsis_arm_conv_partial_fast_opt_q15, METH_VARARGS,""},
{"arm_conv_partial_q31",  cmsis_arm_conv_partial_q31, METH_VARARGS,""},
{"arm_conv_partial_fast_q31",  cmsis_arm_conv_partial_fast_q31, METH_VARARGS,""},
{"arm_conv_partial_opt_q7",  cmsis_arm_conv_partial_opt_q7, METH_VARARGS,""},
{"arm_conv_partial_q7",  cmsis_arm_conv_partial_q7, METH_VARARGS,""},
{"arm_fir_decimate_f32",  cmsis_arm_fir_decimate_f32, METH_VARARGS,""},
{"arm_fir_decimate_init_f32",  cmsis_arm_fir_decimate_init_f32, METH_VARARGS,""},
{"arm_fir_decimate_q15",  cmsis_arm_fir_decimate_q15, METH_VARARGS,""},
{"arm_fir_decimate_fast_q15",  cmsis_arm_fir_decimate_fast_q15, METH_VARARGS,""},
{"arm_fir_decimate_init_q15",  cmsis_arm_fir_decimate_init_q15, METH_VARARGS,""},
{"arm_fir_decimate_q31",  cmsis_arm_fir_decimate_q31, METH_VARARGS,""},
{"arm_fir_decimate_fast_q31",  cmsis_arm_fir_decimate_fast_q31, METH_VARARGS,""},
{"arm_fir_decimate_init_q31",  cmsis_arm_fir_decimate_init_q31, METH_VARARGS,""},
{"arm_fir_interpolate_q15",  cmsis_arm_fir_interpolate_q15, METH_VARARGS,""},
{"arm_fir_interpolate_init_q15",  cmsis_arm_fir_interpolate_init_q15, METH_VARARGS,""},
{"arm_fir_interpolate_q31",  cmsis_arm_fir_interpolate_q31, METH_VARARGS,""},
{"arm_fir_interpolate_init_q31",  cmsis_arm_fir_interpolate_init_q31, METH_VARARGS,""},
{"arm_fir_interpolate_f32",  cmsis_arm_fir_interpolate_f32, METH_VARARGS,""},
{"arm_fir_interpolate_init_f32",  cmsis_arm_fir_interpolate_init_f32, METH_VARARGS,""},
{"arm_biquad_cas_df1_32x64_q31",  cmsis_arm_biquad_cas_df1_32x64_q31, METH_VARARGS,""},
{"arm_biquad_cas_df1_32x64_init_q31",  cmsis_arm_biquad_cas_df1_32x64_init_q31, METH_VARARGS,""},
{"arm_biquad_cascade_df2T_f32",  cmsis_arm_biquad_cascade_df2T_f32, METH_VARARGS,""},
{"arm_biquad_cascade_stereo_df2T_f32",  cmsis_arm_biquad_cascade_stereo_df2T_f32, METH_VARARGS,""},
{"arm_biquad_cascade_df2T_f64",  cmsis_arm_biquad_cascade_df2T_f64, METH_VARARGS,""},
{"arm_biquad_cascade_df2T_init_f32",  cmsis_arm_biquad_cascade_df2T_init_f32, METH_VARARGS,""},
{"arm_biquad_cascade_stereo_df2T_init_f32",  cmsis_arm_biquad_cascade_stereo_df2T_init_f32, METH_VARARGS,""},
{"arm_biquad_cascade_df2T_init_f64",  cmsis_arm_biquad_cascade_df2T_init_f64, METH_VARARGS,""},
{"arm_fir_lattice_init_q15",  cmsis_arm_fir_lattice_init_q15, METH_VARARGS,""},
{"arm_fir_lattice_q15",  cmsis_arm_fir_lattice_q15, METH_VARARGS,""},
{"arm_fir_lattice_init_q31",  cmsis_arm_fir_lattice_init_q31, METH_VARARGS,""},
{"arm_fir_lattice_q31",  cmsis_arm_fir_lattice_q31, METH_VARARGS,""},
{"arm_fir_lattice_init_f32",  cmsis_arm_fir_lattice_init_f32, METH_VARARGS,""},
{"arm_fir_lattice_f32",  cmsis_arm_fir_lattice_f32, METH_VARARGS,""},
{"arm_iir_lattice_f32",  cmsis_arm_iir_lattice_f32, METH_VARARGS,""},
{"arm_iir_lattice_init_f32",  cmsis_arm_iir_lattice_init_f32, METH_VARARGS,""},
{"arm_iir_lattice_q31",  cmsis_arm_iir_lattice_q31, METH_VARARGS,""},
{"arm_iir_lattice_init_q31",  cmsis_arm_iir_lattice_init_q31, METH_VARARGS,""},
{"arm_iir_lattice_q15",  cmsis_arm_iir_lattice_q15, METH_VARARGS,""},
{"arm_iir_lattice_init_q15",  cmsis_arm_iir_lattice_init_q15, METH_VARARGS,""},
{"arm_lms_f32",  cmsis_arm_lms_f32, METH_VARARGS,""},
{"arm_lms_init_f32",  cmsis_arm_lms_init_f32, METH_VARARGS,""},
{"arm_lms_init_q15",  cmsis_arm_lms_init_q15, METH_VARARGS,""},
{"arm_lms_q15",  cmsis_arm_lms_q15, METH_VARARGS,""},
{"arm_lms_q31",  cmsis_arm_lms_q31, METH_VARARGS,""},
{"arm_lms_init_q31",  cmsis_arm_lms_init_q31, METH_VARARGS,""},
{"arm_lms_norm_f32",  cmsis_arm_lms_norm_f32, METH_VARARGS,""},
{"arm_lms_norm_init_f32",  cmsis_arm_lms_norm_init_f32, METH_VARARGS,""},
{"arm_lms_norm_q31",  cmsis_arm_lms_norm_q31, METH_VARARGS,""},
{"arm_lms_norm_init_q31",  cmsis_arm_lms_norm_init_q31, METH_VARARGS,""},
{"arm_lms_norm_q15",  cmsis_arm_lms_norm_q15, METH_VARARGS,""},
{"arm_lms_norm_init_q15",  cmsis_arm_lms_norm_init_q15, METH_VARARGS,""},
{"arm_correlate_f32",  cmsis_arm_correlate_f32, METH_VARARGS,""},
{"arm_correlate_f64",  cmsis_arm_correlate_f64, METH_VARARGS,""},
{"arm_correlate_opt_q15",  cmsis_arm_correlate_opt_q15, METH_VARARGS,""},
{"arm_correlate_q15",  cmsis_arm_correlate_q15, METH_VARARGS,""},
{"arm_correlate_fast_q15",  cmsis_arm_correlate_fast_q15, METH_VARARGS,""},
{"arm_correlate_fast_opt_q15",  cmsis_arm_correlate_fast_opt_q15, METH_VARARGS,""},
{"arm_correlate_q31",  cmsis_arm_correlate_q31, METH_VARARGS,""},
{"arm_correlate_fast_q31",  cmsis_arm_correlate_fast_q31, METH_VARARGS,""},
{"arm_correlate_opt_q7",  cmsis_arm_correlate_opt_q7, METH_VARARGS,""},
{"arm_correlate_q7",  cmsis_arm_correlate_q7, METH_VARARGS,""},
{"arm_fir_sparse_f32",  cmsis_arm_fir_sparse_f32, METH_VARARGS,""},
{"arm_fir_sparse_init_f32",  cmsis_arm_fir_sparse_init_f32, METH_VARARGS,""},
{"arm_fir_sparse_init_q31",  cmsis_arm_fir_sparse_init_q31, METH_VARARGS,""},
{"arm_fir_sparse_init_q15",  cmsis_arm_fir_sparse_init_q15, METH_VARARGS,""},
{"arm_fir_sparse_init_q7",  cmsis_arm_fir_sparse_init_q7, METH_VARARGS,""},

{"arm_circularWrite_f32",  cmsis_arm_circularWrite_f32, METH_VARARGS,""},
{"arm_circularWrite_q15",  cmsis_arm_circularWrite_q15, METH_VARARGS,""},
{"arm_circularWrite_q7",  cmsis_arm_circularWrite_q7, METH_VARARGS,""},

{"arm_levinson_durbin_f32",  cmsis_arm_levinson_durbin_f32, METH_VARARGS,""},
{"arm_levinson_durbin_q31",  cmsis_arm_levinson_durbin_q31, METH_VARARGS,""},


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