###########################################
# Project:      CMSIS DSP Library
# Title:        simu.py
# Description:  Support Python classes for the Python SDF scheduler
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
""" Classes for Python version of the generated schedule
"""

import numpy as np 

class FIFOBase:
    pass 

class FIFO(FIFOBase):
    def __init__(self,fifoSize,fifoBuf):
        self._length = fifoSize
        self._buffer = fifoBuf 
        self._readPos = 0 
        self._writePos = 0 

    def getWriteBuffer(self,nb):

        if (self._readPos > 0):
            self._buffer[:self._writePos-self._readPos] = self._buffer[self._readPos:self._writePos]
            self._writePos = self._writePos - self._readPos
            self._readPos = 0
        
        ret = self._buffer[self._writePos:self._writePos+nb]
        self._writePos = self._writePos + nb
        return(ret)

    def getReadBuffer(self,nb):
        ret = self._buffer[self._readPos:self._readPos+nb]
        self._readPos = self._readPos + nb
        return(ret)

    @property
    def type(self):
        return(self._buffer.dtype)

    def dump(self):
        print(self._buffer)

class GenericNode:
    def __init__(self,inputSize, outputSize,fifoin,fifoout):
        self._src = fifoin
        self._dst = fifoout 
        self._inputSize = inputSize 
        self._outputSize = outputSize

    def getWriteBuffer(self):
        return(self._dst.getWriteBuffer(self._outputSize))

    def getReadBuffer(self):
        return(self._src.getReadBuffer(self._inputSize))

class GenericNode12:
    def __init__(self,inputSize, outputSize1,outputSize2,fifoin,fifoout1,fifoout2):
        self._src = fifoin
        self._dst1 = fifoout1 
        self._dst2 = fifoout2 
        self._inputSize = inputSize 
        self._outputSize1 = outputSize1
        self._outputSize2 = outputSize2


    def getWriteBuffer1(self):
        return(self._dst1.getWriteBuffer(self._outputSize1))

    def getWriteBuffer2(self):
        return(self._dst2.getWriteBuffer(self._outputSize2))

    def getReadBuffer(self):
        return(self._src.getReadBuffer(self._inputSize))

class GenericNode21:
    def __init__(self,inputSize1,inputSize2, outputSize,fifoin1,fifoin2,fifoout):
        self._src1 = fifoin1
        self._src2 = fifoin2
        self._dst = fifoout 
        self._inputSize1 = inputSize1 
        self._inputSize2 = inputSize2 
        self._outputSize = outputSize

    def getWriteBuffer(self):
        return(self._dst.getWriteBuffer(self._outputSize))

    def getReadBuffer1(self):
        return(self._src1.getReadBuffer(self._inputSize1))

    def getReadBuffer2(self):
        return(self._src2.getReadBuffer(self._inputSize2))

class GenericSource:
    def __init__(self, outputSize,fifoout):
        self._dst = fifoout 
        self._outputSize = outputSize

    def getWriteBuffer(self):
        return(self._dst.getWriteBuffer(self._outputSize))

class GenericSink:
    def __init__(self,inputSize,fifoin):
        self._src = fifoin
        self._inputSize = inputSize 
    
    def getReadBuffer(self):
        return(self._src.getReadBuffer(self._inputSize))

class SlidingBuffer(GenericNode):
    def __init__(self,windowSize,overlap,fifoin,fifoout):
        GenericNode.__init__(self,windowSize-overlap,windowSize,fifoin,fifoout)

        self._windowSize = windowSize
        self._overlap = overlap 
        self._memory = np.zeros(overlap)

    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()
        b[:self._overlap] = self._memory 
        b[self._overlap:self._windowSize]=a[:self._windowSize-self._overlap] 
        self._memory[:self._overlap] = b[self._windowSize-self._overlap:self._windowSize]
        
        return(0)


class OverlapTooBig(Exception):
    pass

class OverlapAdd(GenericNode):
    def __init__(self,windowSize,overlap,fifoin,fifoout):
        GenericNode.__init__(self,windowSize,windowSize-overlap,fifoin,fifoout)
        if ((windowSize-overlap)<=0):
           raise OverlapTooBig

        self._windowSize = windowSize
        self._overlap = overlap 
        self._memory = np.zeros(overlap)

    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()

        self._memory[:self._overlap]= a[:self._overlap] + self._memory[:self._overlap] 

        if (2*self._overlap - self._windowSize > 0):
            b[:self._windowSize-self._overlap] = self._memory[:self._windowSize-self._overlap]

            tmp=np.zeros(2*self._overlap - self._windowSize)
            tmp[:] = self._memory[self._windowSize-self._overlap:self._overlap]
            self._memory[:2*self._overlap - self._windowSize] = tmp

            self._memory[2*self._overlap - self._windowSize:self._overlap] = a[self._overlap:self._windowSize]

        elif (2*self._overlap - self._windowSize < 0):
            b[:self._overlap] = self._memory[:self._overlap]
            b[self._overlap:self._windowSize-self._overlap] = a[self._overlap:self._windowSize-self._overlap]
            self._memory[:self._overlap] = a[self._windowSize-self._overlap:self._windowSize]
        else:
            b[:self._overlap]=self._memory[:self._overlap]
            self._memory[:self._overlap] = a[self._overlap:2*self._overlap]

        return(0)
