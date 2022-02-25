###########################################
# Project:      CMSIS DSP Library
# Title:        WavSource.py
# Description:  Source node for reading wave files
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

# It is assuming the input is a stereo file
# It is not yet customizable in this version
# Pad with zero when end of file is reached
class WavSource(GenericSource):
    "Read a stereo wav with 16 bits encoding"
    def __init__(self,outputSize,fifoout,stereo,name):
        GenericSource.__init__(self,outputSize,fifoout)
        self._file=wave.open(name, 'rb')
        self._stereo=stereo
        #print(self._file.getnchannels())
        #print(self._file.getnframes())


    def run(self):
        a=self.getWriteBuffer()

        if self._stereo:
           # Stereo file so chunk must be divided by 2
           frame=np.frombuffer(self._file.readframes(self._outputSize//2),dtype=np.int16)
        else:
           frame=np.frombuffer(self._file.readframes(self._outputSize),dtype=np.int16)
        if frame.size > 0:
           a[:frame.size] = frame
           a[frame.size:] = 0
           return(0)
        else:
           a[:]=0
           return(0)

    def __del__(self):
        self._file.close()