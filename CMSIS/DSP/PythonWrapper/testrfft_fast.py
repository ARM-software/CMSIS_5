import cmsisdsp as dsp
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.fft


def chop(A, eps = 1e-6):
    B = np.copy(A)
    B[np.abs(A) < eps] = 0
    return B

nb = 32
signal = np.cos(2 * np.pi * np.arange(nb) / nb)*np.cos(0.2*2 * np.pi * np.arange(nb) / nb)

#print("{")
#for x in signal:
#  print("%f," % x)
#print("}")

result1=scipy.fft.rfft(signal)
print(chop(result1))
rfftf32=dsp.arm_rfft_fast_instance_f32()
status=dsp.arm_rfft_fast_init_f32(rfftf32,nb)
print(status)
resultI = dsp.arm_rfft_fast_f32(rfftf32,signal,0)
print(chop(resultI))