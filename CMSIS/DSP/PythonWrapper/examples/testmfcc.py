import cmsisdsp as dsp
import numpy as np
import cmsisdsp.fixedpoint as f
import cmsisdsp.mfcc as mfcc
import scipy.signal as sig
from mfccdebugdata import *
from cmsisdsp.datatype import F32


mfccf32=dsp.arm_mfcc_instance_f32()

sample_rate = 16000
FFTSize = 256
numOfDctOutputs = 13
    
freq_min = 64
freq_high = sample_rate / 2
numOfMelFilters = 20

window = sig.hamming(FFTSize, sym=False)
      
filtLen,filtPos,packedFilters = mfcc.melFilterMatrix(F32,freq_min, freq_high, numOfMelFilters,sample_rate,FFTSize)

       
dctMatrixFilters = mfcc.dctMatrix(F32,numOfDctOutputs, numOfMelFilters)




status=dsp.arm_mfcc_init_f32(mfccf32,FFTSize,numOfMelFilters,numOfDctOutputs,dctMatrixFilters,
    filtPos,filtLen,packedFilters,window)
print(status)

tmp=np.zeros(FFTSize + 2)

res=dsp.arm_mfcc_f32(mfccf32,debug,tmp)

print(res)

print(ref)

print(mfccf32.fftLen())
print(mfccf32.nbMelFilters())
print(mfccf32.nbDctOutputs())
