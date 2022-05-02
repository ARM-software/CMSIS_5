import sched as s 
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

import cmsisdsp as dsp
import numpy as np
import cmsisdsp.fixedpoint as f
import cmsisdsp.mfcc as mfcc
import scipy.signal as sig
from cmsisdsp.datatype import F32,Q31,Q15
from sharedconfig import *
import cmsisdsp.datatype as dt

mfccq15=dsp.arm_mfcc_instance_q15()

windowQ15 = dt.convert(sig.hamming(FFTSize, sym=False),Q15)
filtLen,filtPos,packedFiltersQ15 = mfcc.melFilterMatrix(Q15,freq_min, freq_high, numOfMelFilters,sample_rate,FFTSize)
dctMatrixFiltersQ15 = mfcc.dctMatrix(Q15,numOfDctOutputs, numOfMelFilters)


status=dsp.arm_mfcc_init_q15(mfccq15,FFTSize,numOfMelFilters,numOfDctOutputs,
    dctMatrixFiltersQ15,
    filtPos,filtLen,packedFiltersQ15,windowQ15)

#DISPBUF = np.zeros(nbMFCCOutputs*numOfDctOutputs)
DISPBUF=[]
print("Start")

nb,error = s.scheduler(mfccq15,DISPBUF)

print("Nb sched = %d" % nb)


fig, ax = plt.subplots()

# The test signal is 5 second long
# MFCC are slided by 0.5 second by the last window
# Each sink entry is a one second of MFCC
ims = []
for i in range(10):
    mfccdata = (1<<8)*f.Q15toF32(DISPBUF[i])
    mfccdata=mfccdata.reshape((nbMFCCOutputs,numOfDctOutputs))
    mfccdata= np.swapaxes(mfccdata, 0 ,1)
    if i==0:
        ax.imshow(mfccdata, vmin=-10, vmax=10,interpolation='nearest',cmap=cm.coolwarm ,origin='lower')
    
    im=ax.imshow(mfccdata, vmin=-10, vmax=10,interpolation='nearest', animated=True,cmap=cm.coolwarm ,origin='lower')
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=500)

#ani.save("mfcc.mp4")

plt.show()
