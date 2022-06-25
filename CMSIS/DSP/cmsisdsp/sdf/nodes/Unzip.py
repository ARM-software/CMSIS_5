###########################################
# Project:      CMSIS DSP Library
# Title:        Unzip.py
# Description:  Unzip streams
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

# Convert a stream of a1 b1 a2 b2 ...
# to two streams
# a1 a2 ... 
# b1 b2 ....
class Unzip(GenericNode): 
    def __init__(self,inputSize,outputSize1,outputSize2, fifoin,fifoout1,fifoout2):
        GenericNode12.__init__(self,inputSize,outputSize1,outputSize2,fifoin,fifoout1,fifoout2)

    def run(self):
        a=self.getReadBuffer()
        b1=self.getWriteBuffer1()
        b2=self.getWriteBuffer2()
        b1[:]=a[::2]
        b2[:]=a[1::2]
        return(0)