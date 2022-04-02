###########################################
# Project:      CMSIS DSP Library
# Title:        NumpySink.py
# Description:  Sink node for displaying a buffer in scipy
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

from ..simu import *

# Add each new received buffer to a list of buffers
class NumpySink(GenericSink):
    def __init__(self,inputSize,fifoin,matplotlibBuffer):
        GenericSink.__init__(self,inputSize,fifoin)
        self._bufPos=0
        self._buffer=matplotlibBuffer

    def run(self):
        b=self.getReadBuffer()
        buf = np.zeros(self._inputSize)
        buf[:] = b[:]
        self._buffer.append(buf)
       
        return(0)

    