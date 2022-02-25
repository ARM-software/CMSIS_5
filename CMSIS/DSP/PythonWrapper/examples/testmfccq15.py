import cmsisdsp as dsp
import numpy as np
import cmsisdsp.fixedpoint as f
import cmsisdsp.mfcc as mfcc
import scipy.signal as sig
from mfccdebugdata import *
from cmsisdsp.datatype import Q15
import cmsisdsp.datatype as dt

mfccq15=dsp.arm_mfcc_instance_q15()

sample_rate = 16000
FFTSize = 256
numOfDctOutputs = 13
    
freq_min = 64
freq_high = sample_rate / 2
numOfMelFilters = 20

windowQ15 = dt.convert(sig.hamming(FFTSize, sym=False),Q15)
filtLen,filtPos,packedFiltersQ15 = mfcc.melFilterMatrix(Q15,freq_min, freq_high, numOfMelFilters,sample_rate,FFTSize)
dctMatrixFiltersQ15 = mfcc.dctMatrix(Q15,numOfDctOutputs, numOfMelFilters)


status=dsp.arm_mfcc_init_q15(mfccq15,FFTSize,numOfMelFilters,numOfDctOutputs,
    dctMatrixFiltersQ15,
    filtPos,filtLen,packedFiltersQ15,windowQ15)
print("Init status = %d" % status)

tmp=np.zeros(2*FFTSize,dtype=np.int32)

debugQ15 = f.toQ15(debug)
errorStatus,resQ15=dsp.arm_mfcc_q15(mfccq15,debugQ15,tmp)
print("MFCC status = %d" % errorStatus)
res=(1<<8)*f.Q15toF32(resQ15)

print(res)

print(ref)

print("FFT Length = %d" % mfccq15.fftLen())
print("Nb MEL Filters = %d" % mfccq15.nbMelFilters())
print("Nb DCT Outputs = %d" % mfccq15.nbDctOutputs())
