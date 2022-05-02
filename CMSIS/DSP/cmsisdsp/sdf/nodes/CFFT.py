###########################################
# Project:      CMSIS DSP Library
# Title:        CFFTF.py
# Description:  Node for CMSIS-DSP cfft
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



# The CMSIS-DSP CFFT
class CFFT(GenericNode):
    def __init__(self,inputSize,outSize,fifoin,fifoout):
        GenericNode.__init__(self,inputSize,outSize,fifoin,fifoout)
        if fifoin.type == np.dtype(np.float32):
           self._cfft=dsp.arm_cfft_instance_f32()
           status=dsp.arm_cfft_init_f32(self._cfft,inputSize>>1)
        if fifoin.type == np.dtype(np.int16):
           self._cfft=dsp.arm_cfft_instance_q15()
           status=dsp.arm_cfft_init_q15(self._cfft,inputSize>>1)

    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()
        # Copy arrays (not just assign references)
        b[:]=a[:]
        if self._src.type == np.dtype(np.float32):
           dsp.arm_cfft_f32(self._cfft,b,0,1)
        if self._src.type == np.dtype(np.int16):
           dsp.arm_cfft_q15(self._cfft,b,0,1)
        return(0)