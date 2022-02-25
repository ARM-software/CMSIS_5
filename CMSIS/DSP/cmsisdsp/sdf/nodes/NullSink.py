###########################################
# Project:      CMSIS DSP Library
# Title:        NullSink.py
# Description:  Null sink doing nothing for debug
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

# This node is doing nothing
# and can be useful when debugging a graph
class NullSink(GenericSink):
    def __init__(self,inputSize,fifoin,name,matplotlibBuffer):
        GenericSink.__init__(self,inputSize,fifoin)
        

    def run(self):
        # The null sink must at least get a buffer from the FIFO
        # or the FIFO will never be emptied
        # and the scheduling will fail
        i=self.getReadBuffer()
        return(0)