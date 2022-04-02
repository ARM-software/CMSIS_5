###########################################
# Project:      CMSIS DSP Library
# Title:        MFCC.py
# Description:  Node for CMSIS-DSP MFCC
# 
# $Date:        30 July 2021
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
from .simu import *
import cmsisdsp as dsp 



# The CMSIS-DSP MFCC
class MFCC(GenericNode):
    def __init__(self,inputSize,outSize,fifoin,fifoout,mfccConfig):
        GenericNode.__init__(self,inputSize,outSize,fifoin,fifoout)
        self._config=mfccConfig
        if self._src.type == np.dtype(np.float32):
           self._tmp=np.zeros(2*inputSize,dtype=np.float32)
        else:
           self._tmp=np.zeros(2*inputSize,dtype=np.int32)


    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()
        if self._src.type == np.dtype(np.float32):
           res=dsp.arm_mfcc_f32(self._config,a,self._tmp)
           errorStatus = 0
        if self._src.type == np.dtype(np.int32):
           errorStatus,res=dsp.arm_mfcc_q31(self._config,a,self._tmp)
        if self._src.type == np.dtype(np.int16):
           errorStatus,res=dsp.arm_mfcc_q15(self._config,a,self._tmp)
        b[:] = res[:]
        return(errorStatus)