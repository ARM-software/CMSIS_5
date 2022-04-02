###########################################
# Project:      CMSIS DSP Library
# Title:        types.py
# Description:  Description of the basic CMSIS-DSP types
# 
# $Date:        29 July 2021
# $Revision:    V1.10.0
# 
# Target Processor: Cortex-M and Cortex-A cores
# -------------------------------------------------------------------- */
# 
# Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
# 
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
############################################
"""Description of the basic types recognized by the tool"""

F64    = 1
F32    = 2
F16    = 3
Q31    = 4
Q15    = 5
Q7     = 6
UINT32 = 7
UINT16 = 8
UINT8  = 9
SINT32 = 10
SINT16 = 11
SINT8  = 12

class SDFType:
    """Descrition of a type used to fill a FIFO"""

    def __eq__(self, other):
        """Check if this type is equal to another type.
           Two types are equal if they have same class.

           Size differences are handled by the FIFOs ...
        """
        return(self.__class__ == other.__class__)

    @property
    def bytes(self):
       return(0)

class CType(SDFType):
    """A C Scalar element"""
    def __init__(self,typeid):
        self._id=typeid 

    @property
    def bytes(self):
        if self._id == F64:
           return(8)
        elif self._id == F32:
           return(4)
        elif self._id == F16:
           return(2)
        elif self._id == Q31:
           return(7)
        elif self._id == Q15:
           return(2)
        elif self._id == Q7:
           return(1)
        elif self._id == UINT32:
           return(4)
        elif self._id == UINT16:
           return(2)
        elif self._id == UINT8:
           return(1)
        elif self._id == SINT32:
           return(4)
        elif self._id == SINT16:
           return(2)
        elif self._id == SINT8:
           return(1)
        else:
           return(0)

    @property
    def ctype(self):
        if self._id == F64:
           return("float64_t")
        elif self._id == F32:
           return("float32_t")
        elif self._id == F16:
           return("float16_t")
        elif self._id == Q31:
           return("q31_t")
        elif self._id == Q15:
           return("q15_t")
        elif self._id == Q7:
           return("q7_t")
        elif self._id == UINT32:
           return("uint32_t")
        elif self._id == UINT16:
           return("uint16_t")
        elif self._id == UINT8:
           return("uint8_t")
        elif self._id == SINT32:
           return("int32_t")
        elif self._id == SINT16:
           return("int16_t")
        elif self._id == SINT8:
           return("int8_t")
        else:
           return("void")

    @property
    def nptype(self):
        if self._id == F64:
           return("float64")
        elif self._id == F32:
           return("float32")
        elif self._id == F16:
           return("float16")
        elif self._id == Q31:
           return("int32")
        elif self._id == Q15:
           return("int16")
        elif self._id == Q7:
           return("int8")
        elif self._id == UINT32:
           return("uint32")
        elif self._id == UINT16:
           return("uint16")
        elif self._id == UINT8:
           return("uint8")
        elif self._id == SINT32:
           return("int32")
        elif self._id == SINT16:
           return("int16")
        elif self._id == SINT8:
           return("int8")
        else:
           return("void")

    @property
    def graphViztype(self):
        if self._id == F64:
           return("f64")
        elif self._id == F32:
           return("f32")
        elif self._id == F16:
           return("f16")
        elif self._id == Q31:
           return("q31")
        elif self._id == Q15:
           return("q15")
        elif self._id == Q7:
           return("q7")
        elif self._id == UINT32:
           return("u32")
        elif self._id == UINT16:
           return("u16")
        elif self._id == UINT8:
           return("u8")
        elif self._id == SINT32:
           return("s32")
        elif self._id == SINT16:
           return("s16")
        elif self._id == SINT8:
           return("s8")
        else:
           return("void")

    @property
    def dspExtension(self):
        if self._id == F64:
           return("f64")
        elif self._id == F32:
           return("f32")
        elif self._id == F16:
           return("f16")
        elif self._id == Q31:
           return("q31")
        elif self._id == Q15:
           return("q15")
        elif self._id == Q7:
           return("q7")
        elif self._id == UINT32:
           return("u32")
        elif self._id == UINT16:
           return("u16")
        elif self._id == UINT8:
           return("u8")
        elif self._id == SINT32:
           return("s32")
        elif self._id == SINT16:
           return("s16")
        elif self._id == SINT8:
           return("s8")
        else:
           return("void")

    def __eq__(self, other):
      return(SDFType.__eq__(self,other) and self._id == other._id)


