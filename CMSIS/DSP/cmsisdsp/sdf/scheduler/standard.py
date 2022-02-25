###########################################
# Project:      CMSIS DSP Library
# Title:        standard.py
# Description:  Standard nodes to describe a network
# 
# $Date:        02 August 2021
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
"""Standard nodes available to describe a network in addition to the generic nodes"""

from ..types import *
from .node import GenericNode,GenericSource,GenericSink

floatType=CType(F32)

class Unzip(GenericNode):

    def __init__(self,name,theType,length):
        GenericNode.__init__(self,name)
        self._length = length 
        self.addInput("i",theType,2*length)
        self.addOutput("o1",theType,length)
        self.addOutput("o2",theType,length)
    
    @property
    def typeName(self):
        return "Unzip"

class Zip(GenericNode):

    def __init__(self,name,theType,length):
        GenericNode.__init__(self,name)
        self._length = length 
        self.addInput("i1",theType,length)
        self.addInput("i2",theType,length)
        self.addOutput("o",theType,length)
    
    @property
    def typeName(self):
        return "Zip"



class CFFT(GenericNode):
    def __init__(self,name,theType,inLength):
        GenericNode.__init__(self,name)

        self.addInput("i",theType,2*inLength)
        self.addOutput("o",theType,2*inLength)

    @property
    def typeName(self):
        return "CFFT"

class ICFFT(GenericNode):
    def __init__(self,name,theType,inLength):
        GenericNode.__init__(self,name)

        self.addInput("i",theType,2*inLength)
        self.addOutput("o",theType,2*inLength)

    @property
    def typeName(self):
        return "ICFFT"

class ToComplex(GenericNode):
    def __init__(self,name,theType,inLength):
        GenericNode.__init__(self,name)

        self.addInput("i",theType,inLength)
        self.addOutput("o",theType,2*inLength)

    @property
    def typeName(self):
        return "ToComplex"

class ToReal(GenericNode):
    def __init__(self,name,theType,inLength):
        GenericNode.__init__(self,name)

        self.addInput("i",theType,2*inLength)
        self.addOutput("o",theType,inLength)

    @property
    def typeName(self):
        return "ToReal"


class NullSink(GenericSink):
    def __init__(self,name,theType,inLength):
        GenericSink.__init__(self,name)
        self.addInput("i",theType,inLength)

    @property
    def typeName(self):
        return "NullSink"

class StereoToMono(GenericNode):
    def __init__(self,name,theType,outLength):
        GenericNode.__init__(self,name)
        self.addInput("i",theType,2*outLength)
        self.addOutput("o",theType,outLength)

    @property
    def typeName(self):
        return "StereoToMono"


class MFCC(GenericNode):
    def __init__(self,name,theType,inLength,outLength):
        GenericNode.__init__(self,name)

        self.addInput("i",theType,inLength)
        self.addOutput("o",theType,outLength)

    @property
    def typeName(self):
        return "MFCC"

#############################
#
# Host only Nodes
#

class FileSource(GenericSource):
    def __init__(self,name,inLength):
        GenericSource.__init__(self,name)
        floatType=CType(F32)
        self.addOutput("o",floatType,inLength)

    @property
    def typeName(self):
        return "FileSource"

class FileSink(GenericSink):
    def __init__(self,name,inLength):
        GenericSink.__init__(self,name)
        floatType=CType(F32)
        self.addInput("i",floatType,inLength)

    @property
    def typeName(self):
        return "FileSink"

#############################
#
# Python and host only Nodes
#


class WavSource(GenericSource):
    def __init__(self,name,inLength):
        GenericSource.__init__(self,name)
        q15Type=CType(Q15)
        self.addOutput("o",q15Type,inLength)

    @property
    def typeName(self):
        return "WavSource"

class WavSink(GenericSink):
    def __init__(self,name,inLength):
        GenericSink.__init__(self,name)
        q15Type=CType(Q15)
        self.addInput("i",q15Type,inLength)

    @property
    def typeName(self):
        return "WavSink"

class NumpySink(GenericSink):
    def __init__(self,name,theType,inLength):
        GenericSink.__init__(self,name)
        self.addInput("i",theType,inLength)

    @property
    def typeName(self):
        return "NumpySink"

##################
#
# Node to communicates with a VHT block running in Modelica
#
# It is requiring the VHT Modelica extensions which can be found
# in the VHTSystemModeling repository on ArmSoftware GitHub

class VHTSource(GenericNode):
    def __init__(self,name,inLength,theID):
        GenericSource.__init__(self,name)

        self.addOutput("o",CType(Q15),inLength)
        self.addLiteralArg(theID)


    @property
    def typeName(self):
        return "VHTSource"

class VHTSink(GenericNode):
    def __init__(self,name,inLength,theID):
        GenericSource.__init__(self,name)

        self.addInput("i",CType(Q15),inLength)
        self.addLiteralArg(theID)

    @property
    def typeName(self):
        return "VHTSink"
        