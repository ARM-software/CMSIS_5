###########################################
# Project:      CMSIS DSP Library
# Title:        FileSource.py
# Description:  Node for creating file source
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

# Read a list of float from a file
# and pad with 0 indefinitely when  end of file is reached.
class FileSource(GenericSource):
    def __init__(self,outputSize,fifoout,name):
        GenericSource.__init__(self,outputSize,fifoout)
        self._file=open(name,"r")

    def run(self):
        a=self.getWriteBuffer()

        for i in range(self._outputSize):
            s=self._file.readline()
            if (len(s)>0):
               a[i]=float(s)
            else:
               a[i] = 0
        return(0)

    def __del__(self):
        self._file.close()