import cmsisdsp as dsp
import numpy as np
import cmsisdsp.fixedpoint as f
import cmsisdsp.mfcc as mfcc
import scipy.signal as sig
from mfccdebugdata import *
from cmsisdsp.datatype import Q31
import cmsisdsp.datatype as dt

mfccq31=dsp.arm_mfcc_instance_q31()

sample_rate = 16000
FFTSize = 256
numOfDctOutputs = 13
    
freq_min = 64
freq_high = sample_rate / 2
numOfMelFilters = 20

windowQ31 = dt.convert(sig.hamming(FFTSize, sym=False),Q31)
filtLen,filtPos,packedFiltersQ31 = mfcc.melFilterMatrix(Q31,freq_min, freq_high, numOfMelFilters,sample_rate,FFTSize)
dctMatrixFiltersQ31 = mfcc.dctMatrix(Q31,numOfDctOutputs, numOfMelFilters)


status=dsp.arm_mfcc_init_q31(mfccq31,FFTSize,numOfMelFilters,numOfDctOutputs,
    dctMatrixFiltersQ31,
    filtPos,filtLen,packedFiltersQ31,windowQ31)
print("Init status = %d" % status)

tmp=np.zeros(2*FFTSize,dtype=np.int32)

debugQ31 = f.toQ31(debug)
errorStatus,resQ31=dsp.arm_mfcc_q31(mfccq31,debugQ31,tmp)
print("MFCC status = %d" % errorStatus)
res=(1<<8)*f.Q31toF32(resQ31)

print(res)

print(ref)

print("FFT Length = %d" % mfccq31.fftLen())
print("Nb MEL Filters = %d" % mfccq31.nbMelFilters())
print("Nb DCT Outputs = %d" % mfccq31.nbDctOutputs())
