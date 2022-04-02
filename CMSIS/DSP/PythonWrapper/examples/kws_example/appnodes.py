###########################################
# Project:      CMSIS DSP Library
# Title:        appnodes.py
# Description:  Application nodes for kws example
# 
# $Date:        16 March 2022
# $Revision:    V1.10.0
# 
# Target Processor: Cortex-M and Cortex-A cores
# -------------------------------------------------------------------- */
# 
# Copyright (C) 2010-2022 ARM Limited or its affiliates. All rights reserved.
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
from cmsisdsp.sdf.nodes.simu import *
from custom import *
import cmsisdsp.fixedpoint as fix
import cmsisdsp as dsp 


# Sink node displaying the recognized word
class Sink(GenericSink):
    def __init__(self,inputSize,fifoin):
        GenericSink.__init__(self,inputSize,fifoin)
        

    def run(self):
        i=self.getReadBuffer()

        if i[0] == -1:
            print("Unknown")

        if i[0] == 0:
            print("Yes")

        return(0)

# Source node getting sample from NumPy buffer
# At the end of the buffer we generate 0 samples
class Source(GenericSource):
    def __init__(self,outputSize,fifoout,buffer):
        GenericSource.__init__(self,outputSize,fifoout)
        self._offset=0
        self._buffer=np.array(buffer)

    def run(self):
        a=self.getWriteBuffer()

        if self._offset + self._outputSize >= len(self._buffer):
            a[0:self._outputSize] = np.zeros(self._outputSize,dtype=np.int16)
        else:
           a[0:self._outputSize] = self._buffer[self._offset:self._offset+self._outputSize]
           self._offset = self._offset + self._outputSize

        return(0)


# Same implementation as the one in the python notebook
def dsp_zcr_q15(w):
    m = dsp.arm_mean_q15(w)
    # Negate can saturate so we use CMSIS-DSP function which is working on array (and we have a scalar)
    m = dsp.arm_negate_q15(np.array([m]))[0]
    w = dsp.arm_offset_q15(w,m)
    
    f=w[:-1]
    g=w[1:]
    k=np.count_nonzero(np.logical_and(np.logical_or(np.logical_and(f>0,g<0), np.logical_and(f<0,g>0)),g>f))
    
    # k < len(f) so shift should be 0 except when k == len(f)
    # When k==len(f) normally quotient is 0x4000 and shift 1 and we convert
    # this to 0x7FFF
    status,quotient,shift_val=dsp.arm_divide_q15(k,len(f))
    if shift_val==1:
        return(dsp.arm_shift_q31(np.array([quotient]),shift)[0])
    else:
        return(quotient)

# Same implementation as the one in the notebook
class FIR(GenericNode):
    def __init__(self,inputSize,outSize,fifoin,fifoout):
        GenericNode.__init__(self,inputSize,outSize,fifoin,fifoout)
        self._firq15=dsp.arm_fir_instance_q15()

        
        

    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()
        errorStatus = 0

        blockSize=self._inputSize
        numTaps=10
        stateLength = numTaps + blockSize - 1
        dsp.arm_fir_init_q15(self._firq15,10,fix.toQ15(np.ones(10)/10.0),np.zeros(stateLength,dtype=np.int16))

        b[:] = dsp.arm_fir_q15(self._firq15,a)
        
        return(errorStatus)

class Feature(GenericNode):
    def __init__(self,inputSize,outSize,fifoin,fifoout,window):
        GenericNode.__init__(self,inputSize,outSize,fifoin,fifoout)
        self._window=window
        

    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()
        errorStatus = 0

        b[:] = dsp_zcr_q15(dsp.arm_mult_q15(a,self._window))

        return(errorStatus)


    
class KWS(GenericNode):
    def __init__(self,inputSize,outSize,fifoin,fifoout,coef_q15,coef_shift,intercept_q15,intercept_shift):
        GenericNode.__init__(self,inputSize,outSize,fifoin,fifoout)
        self._coef_q15=coef_q15
        self._coef_shift=coef_shift
        self._intercept_q15=intercept_q15
        self._intercept_shift=intercept_shift
        

    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()
        errorStatus = 0

        res=dsp.arm_dot_prod_q15(self._coef_q15,a)
    
        scaled=dsp.arm_shift_q15(np.array([self._intercept_q15]),self._intercept_shift-self._coef_shift)[0]
        # Because dot prod output is in Q34.30
        # and ret is on 64 bits
        scaled = np.int64(scaled) << 15 
    
        res = res + scaled
    
   
    
        if res<0:
            b[0]=-1
        else:
            b[0]=0
        
        return(errorStatus)

