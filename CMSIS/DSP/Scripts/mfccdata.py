###########################################
# Project:      CMSIS DSP Library
# Title:        mfccdata.py
# Description:  Generation of MFCC arays for the MFCC C init function
# 
# $Date:        07 September 2021
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
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape,FileSystemLoader
import os.path
import struct
import scipy.signal as sig

def to_q31(v):
    r = int(round(v * 2**31))
    if (r > 0x07FFFFFFF):
      r = 0x07FFFFFFF
    if (r < -0x080000000):
      r = -0x080000000
    return ("0x%s" % format(struct.unpack('<I', struct.pack('<i', r))[0],'08X'))

def to_q15(v):
    r = int(round(v * 2**15))
    if (r > 0x07FFF):
      r = 0x07FFF
    if (r < -0x08000):
      r = -0x08000
    return ("0x%s" % format(struct.unpack('<H', struct.pack('<h', r))[0],'04X'))

def to_f16(v):
    return("(float16_t)%ff" % struct.unpack('<f',struct.pack('<f',v)))

def to_f32(v):
     return("%ff" % struct.unpack('<f',struct.pack('<f',v)))

class ConvertArray:
    def __init__(self,theType):
        self._cvt = lambda x : x
        if theType=="f32":
            self._cvt = to_f32
        if theType=="f16":
            self._cvt = to_f16 
        if theType=="q31":
            self._cvt = to_q31
        if theType=="q15":
            self._cvt = to_q15

    def getArrayContent(self,samples):
        nb = 0
        res=""
        res += "{\n"
        for sample in samples:
            res += str(self._cvt(sample))
            res += ","
            nb = nb + 1 
            if nb == 10:
                res += "\n"
                nb = 0 
        res += "}"
        return(res)



def frequencyToMelSpace(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)

def melSpaceToFrequency(mels):
    return 700.0 * (np.exp(mels / 1127.0) - 1.0)

def melFilterMatrix(fmin, fmax, numOfMelFilters,fs,FFTSize):

    filters = np.zeros((numOfMelFilters,int(FFTSize/2+1)))
    zeros = np.zeros(int(FFTSize // 2 ))


    fmin_mel = frequencyToMelSpace(fmin)
    fmax_mel = frequencyToMelSpace(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=numOfMelFilters+2)


    linearfreqs = np.linspace( 0, fs/2.0, int(FFTSize // 2 + 1) )
    spectrogrammels = frequencyToMelSpace(linearfreqs)[1:]


    filtPos=[]
    filtLen=[]
    totalLen = 0
    packedFilters = []
    for n in range(numOfMelFilters):

      
      upper = (spectrogrammels - mels[n])/(mels[n+1]-mels[n]) 
      lower = (mels[n+2] - spectrogrammels)/(mels[n+2]-mels[n+1])


      filters[n, :] = np.hstack([0,np.maximum(zeros,np.minimum(upper,lower))])
      nb = 0 
      startFound = False
      for sample in filters[n, :]:
        if not startFound and sample != 0.0:
            startFound = True 
            startPos = nb

        if startFound and sample == 0.0:
           endPos = nb - 1 
           break
        nb = nb + 1 
      filtLen.append(endPos - startPos+1)
      totalLen += endPos - startPos + 1
      filtPos.append(startPos)
      packedFilters += list(filters[n, startPos:endPos+1])

    return filtLen,filtPos,totalLen,packedFilters,filters


def dctMatrix(numOfDctOutputs, numOfMelFilters):
   
    result = np.zeros((numOfDctOutputs,numOfMelFilters))
    s=(np.linspace(1,numOfMelFilters,numOfMelFilters) - 0.5)/numOfMelFilters

    for i in range(0, numOfDctOutputs):
        result[i,:]=np.cos(i * np.pi*s) * np.sqrt(2.0/numOfMelFilters)
        
    return result


def ctype(s):
    if s == "f64":
        return("float64_t")
    if s == "f32":
        return("float32_t")
    if s == "f16":
        return("float16_t")
    if s == "q31":
        return("q31_t")
    if s == "q15":
        return("q15_t")

def typeext(s):
    if s == "f64":
        return("_f64")
    if s == "f32":
        return("_f32")
    if s == "f16":
        return("_f16")
    if s == "q31":
        return("_q31") 
    if s == "q15":
        return("_q15")

def prepareWindowConfig(configs):
    # sig.hamming(FFTSize, sym=False) 
    for config in configs:
        c=configs[config] 
        if c["win"] == "hamming":
           win = sig.hamming(c["fftlength"], sym=False) 
        if c["win"] == "hanning":
           win = sig.hann(c["fftlength"], sym=False) 

        cvt=ConvertArray(c["type"])
        c["ctype"]=ctype(c["type"])
        c["ext"]=typeext(c["type"])

        c["winSamples"] = cvt.getArrayContent(win)

def prepareMelconfig(configs):
    for config in configs:
        c=configs[config]

        cvt=ConvertArray(c["type"])
        cvtInt=ConvertArray(None)
        c["ctype"]=ctype(c["type"])
        c["ext"]=typeext(c["type"])

        filtLen,filtPos,totalLen,packedFilters,filters = melFilterMatrix(c["fmin"], c["fmax"], c["melFilters"],c["samplingRate"],c["fftlength"])
    
        c["filtLenArray"]=cvtInt.getArrayContent(filtLen)
        c["filtPosArray"]=cvtInt.getArrayContent(filtPos)
        c["totalLen"]=totalLen
        c["filters"]=cvt.getArrayContent(packedFilters)

def prepareDctconfig(configs):
    for config in configs:
        c=configs[config]

        cvt=ConvertArray(c["type"])
        c["ctype"]=ctype(c["type"])
        c["ext"]=typeext(c["type"])
        c["dctMatrixLength"]=c["dctOutputs"] * c["melFilters"]

        dctMat = dctMatrix(c["dctOutputs"],c["melFilters"])
        dctMat=dctMat.reshape(c["dctMatrixLength"])
        c["dctMatrix"]=cvt.getArrayContent(dctMat)

    #print(configs)

def checkF16(configs):
    hasF16 = False
    for config in configs["dct"]:
        c=configs["dct"][config]
        if c["type"]=="f16":
           hasF16 = True
           c["hasF16"]=True

    for config in configs["melfilter"]:
        c=configs["melfilter"][config]
        if c["type"]=="f16":
           hasF16 = True
           c["hasF16"]=True

    for config in configs["window"]:
        c=configs["window"][config]
        if c["type"]=="f16":
           hasF16 = True
           c["hasF16"]=True

    configs["hasF16"]=hasF16

env = Environment(
       # For 3.0 version of jinja2, replace with
       # loader=PackageLoader("mfcctemplates",""),
       loader=PackageLoader("mfccdata","mfcctemplates"),
       autoescape=select_autoescape(),
       trim_blocks=True
    )
    
ctemplate = env.get_template("mfccdata.c")
htemplate = env.get_template("mfccdata.h")  


def genMfccHeader(f,configs,filename):
    print(htemplate.render(configs=configs,filename=filename),file=f)

def genMfccInit(f,configs,filename):
    print(ctemplate.render(configs=configs,filename=filename),file=f)