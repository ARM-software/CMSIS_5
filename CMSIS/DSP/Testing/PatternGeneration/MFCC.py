###########################################
# Project:      CMSIS DSP Library
# Title:        MFCC.py
# Description:  Test pattern generation for MFCC
# 
# $Date:        02 September 2021
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
import os.path
import numpy as np
import itertools
import Tools
import scipy
import scipy.signal as sig
import scipy.fftpack

################################
#
# Gives the same results as the tensorflow lite
# MFCC if hamming window is used
# (TF stft) is using hanning by default
#

DEBUG = False

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


    for n in range(numOfMelFilters):

      
      upper = (spectrogrammels - mels[n])/(mels[n+1]-mels[n]) 
      lower = (mels[n+2] - spectrogrammels)/(mels[n+2]-mels[n+1])


      filters[n, :] = np.hstack([0,np.maximum(zeros,np.minimum(upper,lower))])

    return filters


def dctMatrix(numOfDctOutputs, numOfMelFilters):
   
    result = np.zeros((numOfDctOutputs,numOfMelFilters))
    s=(np.linspace(1,numOfMelFilters,numOfMelFilters) - 0.5)/numOfMelFilters

    for i in range(0, numOfDctOutputs):
        result[i,:]=np.cos(i * np.pi*s) * np.sqrt(2.0/numOfMelFilters)
        
    return result



class MFCCConfig:
   def __init__(self,freq_min,freq_high,numOfMelFilters,numOfDctOutputs,FFTSize,sample_rate):
      self._freq_min=freq_min
      self._freq_high=freq_high 
      self._numOfMelFilters = numOfMelFilters
      self._FFTSize=FFTSize 
      self._sample_rate=sample_rate
      #self._window = sig.hann(FFTSize, sym=True)
      self._window = sig.hamming(FFTSize, sym=False)
      #print(self._window)
      self._numOfDctOutputs=numOfDctOutputs

      self._filters = melFilterMatrix(freq_min, freq_high, numOfMelFilters,sample_rate,FFTSize)

       
      self._dctMatrixFilters = dctMatrix(numOfDctOutputs, numOfMelFilters)

   def mfcc(self,audio):
       m = np.amax(np.abs(audio))
       if m != 0:
          s = 1.0 / m
       else:
          s = 1.0
       audio = audio * s

       audioWin = audio * self._window

       if DEBUG:
         print(audioWin)

       audioFFT = scipy.fftpack.fft(audioWin)
       if DEBUG:
         print(audioFFT)

       audioPower = np.abs(audioFFT)
       if DEBUG:
         print(audioPower)

       filterLimit = int(1 + self._FFTSize // 2)
       audioPower=audioPower[:filterLimit]
      
       audioFiltered = np.dot(self._filters,audioPower)
       if DEBUG:
         print(audioFiltered)

       audioLog = np.log(audioFiltered + 1e-6)

       cepstral_coefficents = np.dot(self._dctMatrixFilters, audioLog)
   
       return(cepstral_coefficents)


debug=np.array([ 0.65507051 ,-0.94647589 ,0.00627239 ,0.14151286 ,-0.10863318 ,-0.36370327
 ,0.05777126 ,-0.11915792 ,0.50183546 ,-0.31461335 ,0.66440771 ,0.05389963
 ,0.39690544 ,0.25424852 ,-0.17045277 ,0.09649268 ,0.87357385 ,-0.44666372
 ,-0.02637822 ,-0.10055151 ,-0.14610252 ,-0.05981251 ,-0.02999124 ,0.60923213
 ,0.10530095 ,0.35684248 ,0.21845946 ,0.47845017 ,-0.60206979 ,0.25186908
 ,-0.27410056 ,-0.07080467 ,-0.05109539 ,-0.2666572 ,0.25483105 ,-0.86459185
 ,0.07733397 ,-0.58535444 ,0.06230904 ,-0.04161475 ,-0.17467296 ,0.77721125
 ,-0.01728161 ,-0.32141218 ,0.36674466 ,-0.17932843 ,0.78486115 ,0.12469579
 ,-0.94796877 ,0.05536031 ,0.32627676 ,0.46628512 ,-0.02585836 ,-0.51439834
 ,0.21387904 ,0.16319442 ,-0.01020818 ,-0.77161183 ,0.07754634 ,-0.24970455
 ,0.2368003 ,0.35167963 ,0.14620137 ,-0.02415204 ,0.91086167 ,-0.02434647
 ,-0.3968239 ,-0.04703925 ,-0.43905103 ,-0.34834965 ,0.33728158 ,0.15138992
 ,-0.43218885 ,0.26619718 ,0.07177906 ,0.33393581 ,-0.50306915 ,-0.63101084
 ,-0.08128395 ,-0.06569788 ,0.84232797 ,-0.32436751 ,0.02528537 ,-0.3498329
 ,0.41859931 ,0.07794887 ,0.4571989 ,0.24290963 ,0.08437417 ,-0.01371585
 ,-0.00103008 ,0.83978697 ,-0.29001237 ,0.14438743 ,0.11943318 ,-0.25576402
 ,0.25151083 ,0.07886626 ,0.11565781 ,-0.01582203 ,0.1310246 ,-0.5553611
 ,-0.37950665 ,0.44179691 ,0.08460877 ,0.30646419 ,0.48927934 ,-0.21240309
 ,0.36844264 ,0.49686615 ,-0.81617664 ,0.52221472 ,-0.05188992 ,-0.03929655
 ,-0.47674501 ,-0.54506781 ,0.30711148 ,0.10049337 ,-0.47549213 ,0.59106713
 ,-0.62276051 ,-0.35182917 ,0.14612027 ,0.56142168 ,-0.01053732 ,0.35782179
 ,-0.27220781 ,-0.03672346 ,-0.11282222 ,0.3364912 ,-0.22352515 ,-0.04245287
 ,0.56968605 ,-0.14023724 ,-0.82982905 ,0.00860008 ,0.37920345 ,-0.53749318
 ,-0.12761215 ,0.08567603 ,0.47020765 ,-0.28794812 ,-0.33888971 ,0.01850441
 ,0.66848233 ,-0.26532759 ,-0.20777571 ,-0.68342729 ,-0.41498696 ,0.00593224
 ,0.02229368 ,0.75596329 ,0.29447568 ,-0.1106449 ,0.24181939 ,0.05807497
 ,-0.14343857 ,0.304988 ,0.00689148 ,-0.06264758 ,0.25864714 ,-0.22252155
 ,0.28621689 ,0.17031599 ,-0.34694027 ,-0.01625718 ,0.39834181 ,0.01259659
 ,-0.28022716 ,-0.02506168 ,-0.10276881 ,0.31733924 ,0.02787068 ,-0.09824124
 ,0.45147797 ,0.14451518 ,0.17996395 ,-0.70594978 ,-0.92943177 ,0.13649282
 ,-0.5938426 ,0.50289928 ,0.19635269 ,0.16811504 ,0.05803999 ,0.0037204
 ,0.13847419 ,0.30568038 ,0.3700732 ,0.21257548 ,-0.31151753 ,-0.28836886
 ,0.68743932 ,-0.11084429 ,-0.4673766 ,0.16637754 ,-0.38992572 ,0.16505578
 ,-0.07499844 ,0.04226538 ,-0.11042177 ,0.0704542 ,-0.632819 ,-0.54898472
 ,0.26498649 ,-0.59380386 ,0.93387213 ,0.06526726 ,-0.23223558 ,0.07941394
 ,0.14325166 ,0.26914661 ,0.00925575 ,-0.34282161 ,-0.51418231 ,-0.12011075
 ,-0.26676314 ,-0.09999028 ,0.03027513 ,0.22846503 ,-0.08930338 ,-0.1867156
 ,0.66297846 ,0.32220769 ,-0.06015469 ,0.04034043 ,0.09595597 ,-1.
 ,-0.42933352 ,0.25069376 ,-0.26030918 ,-0.28511861 ,-0.19931228 ,0.24408572
 ,-0.3231952 ,0.45688981 ,-0.07354078 ,0.25669449 ,-0.44202722 ,0.11928406
 ,-0.32826109 ,0.52660984 ,0.03067858 ,0.11095242 ,0.19933679 ,0.03042371
 ,-0.34768682 ,0.09108447 ,0.61234556 ,0.1854931 ,0.19680502 ,0.27617564
 ,0.33381827 ,-0.47358967 ,0.28714328 ,-0.27495982])

def noiseSignal(nb):
    return(2.0*np.random.rand(nb)-1.0)

def sineSignal(freqRatio,nb):
    fc = nb / 2.0
    f = freqRatio*fc 
    time = np.arange(0,nb)
    return(np.sin(2 * np.pi * f *  time/nb))

def noisySineSignal(noiseAmp,r,nb):
    return(noiseAmp*noiseSignal(nb) + r*sineSignal(r,nb))

def writeTests(config,format):
    NBSAMPLES=[256,512,1024]
    if DEBUG:
       NBSAMPLES=[256]
    

    sample_rate = 16000
    FFTSize = 256
    numOfDctOutputs = 13
    
    freq_min = 64
    freq_high = sample_rate / 2
    numOfMelFilters = 20

    for nb in NBSAMPLES:
        inputsNoise=[] 
        inputsSine=[] 
        outputsNoise=[] 
        outputsSine=[] 
        inNoiselengths=[]
        outNoiselengths=[]
        inSinelengths=[]
        outSinelengths=[]

        
        FFTSize=nb
        mfccConfig=MFCCConfig(freq_min,freq_high,numOfMelFilters,numOfDctOutputs,FFTSize,sample_rate)
        
        # Add noise
        audio=np.random.randn(nb)
        audio = Tools.normalize(audio)
        if DEBUG:
           audio=debug
        inputsNoise += list(audio)
        refNoise=mfccConfig.mfcc(audio)
        if format == Tools.Q15:
            refNoise = refNoise / (1<<8)
        if format == Tools.Q31:
            refNoise = refNoise / (1<<8)
        #print(audio)
        if DEBUG:
           print(refNoise)
        outputsNoise+=list(refNoise)
        inNoiselengths+=[nb]
        outNoiselengths+=[numOfDctOutputs]

        
        config.writeInput(1, inputsNoise,"MFCCNoiseInput_%d_" % nb)
        config.writeReference(1, outputsNoise,"MFCCNoiseRef_%d_" % nb)

        # Sine
        audio=noisySineSignal(0.1,0.8,nb)
        #audio = Tools.normalize(audio)
        inputsSine += list(audio)
        refSine=mfccConfig.mfcc(audio)
        if format == Tools.Q15:
            refSine = refSine / (1<<8)
        if format == Tools.Q31:
            refSine = refSine / (1<<8)
        #print(audio)
        outputsSine+=list(refSine)
        inSinelengths+=[nb]
        outSinelengths+=[numOfDctOutputs]

        
        config.writeInput(1, inputsSine,"MFCCSineInput_%d_" % nb)
        config.writeReference(1, outputsSine,"MFCCSineRef_%d_" % nb)

    
   

def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Transform","MFCC")
    PARAMDIR = os.path.join("Parameters","DSP","Transform","MFCC")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configf16=Tools.Config(PATTERNDIR,PARAMDIR,"f16")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    #configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")

    configf32.setOverwrite(False)
    configf16.setOverwrite(False)
    configq31.setOverwrite(False)
    configq15.setOverwrite(False)

   
    writeTests(configf32,0)
    writeTests(configf16,Tools.F16)

    writeTests(configq31,Tools.Q31)
    writeTests(configq15,Tools.Q15)
   
if __name__ == '__main__':
  generatePatterns()
