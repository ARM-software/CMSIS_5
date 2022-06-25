###########################################
# Project:      CMSIS DSP Library
# Title:        Zip.py
# Description:  Zip two streams
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


class Unzip(GenericNode): 
    def __init__(self,inputSize1,inputSize2,outputSize, fifoin1,fifoin2,fifoout):
        GenericNode21.__init__(self,inputSize1,inputSize2,outputSize,fifoin1,fifoin2,fifoout)

    def run(self):
        a1=self.getReadBuffer1()
        a2=self.getReadBuffer2()
        b=self.getWriteBuffer()
        b[::2]=a1
        b[1::2]=a2
        return(0)
