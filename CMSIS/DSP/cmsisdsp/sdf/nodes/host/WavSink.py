###########################################
# Project:      CMSIS DSP Library
# Title:        WavSink.py
# Description:  Sink node for creating a wav
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
import wave
import struct 

# Mono with 16 bits encoding
class WavSink(GenericSink):
    def __init__(self,inputSize,fifoin,name,matplotlibBuffer):
        GenericSink.__init__(self,inputSize,fifoin)
        self._bufPos=0
        self._buffer=matplotlibBuffer
        self._file = wave.open(name,'w')
        self._file.setnchannels(1) # mono
        self._file.setsampwidth(2)
        self._file.setframerate(16000)

    def run(self):
        b=self.getReadBuffer()
        nextPos = self._bufPos + self._inputSize 
        if (nextPos <= self._buffer.size):
            # Save output to buffer defined in custom.py 
            # and used to display result with matplotlib
            self._buffer[self._bufPos:self._bufPos+self._inputSize]=b[:]
            self._bufPos = nextPos

        for sample in b:
            data = struct.pack('<h', sample)
            self._file.writeframesraw( data )

        return(0)

    def __del__(self):
        self._file.close()