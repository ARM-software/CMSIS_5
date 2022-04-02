###########################################
# Project:      CMSIS DSP Library
# Title:        StereoToMono.py
# Description:  Stereo to mono in Q15
# 
# $Date:        06 August 2021
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
import numpy as np 
import cmsisdsp as dsp

class StereoToMono(GenericNode):
    def __init__(self,inputSize,outputSize,fifoin,fifoout):
        GenericNode.__init__(self,inputSize,outputSize,fifoin,fifoout)
        if fifoin.type == np.dtype(np.float32):
            self._isFloat=True 
        else:
            self._isFloat=False


    def run(self):
        i=self.getReadBuffer()
        o=self.getWriteBuffer()

        if self._isFloat:
           o[:] = 0.5 * (i[::2] + i[1::2])
        else:
           o[:] = (i[::2]) // 2 + (i[1::2] // 2)
        return(0)