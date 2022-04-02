###########################################
# Project:      CMSIS DSP Library
# Title:        FileSink.py
# Description:  Node for creating file sinks
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
from ..simu import *
import numpy as np 


# Write a list of samples to a file in text form
class FileSink(GenericSink):
    def __init__(self,inputSize,fifoin,name,matplotlibBuffer):
        GenericSink.__init__(self,inputSize,fifoin)
        self._file=open(name,"w")
        self._bufPos=0
        self._buffer=matplotlibBuffer

    def run(self):
        b=self.getReadBuffer()
        nextPos = self._bufPos + self._inputSize 
        if (nextPos <= self._buffer.size):
            # Save output to buffer defined in custom.py 
            # and used to display result with matplotlib
            self._buffer[self._bufPos:self._bufPos+self._inputSize ]=b[:]
            self._bufPos = nextPos

        for sample in b:
            print(sample,file=self._file)

        return(0)


    def __del__(self):
        self._file.close()