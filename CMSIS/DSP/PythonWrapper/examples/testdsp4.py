import cmsisdsp as dsp
import numpy as np
from scipy import signal
from scipy.fftpack import dct 
import cmsisdsp.fixedpoint as f
from pyquaternion import Quaternion

import colorama
from colorama import init,Fore, Back, Style
import statsmodels.tsa.stattools

import scipy.spatial


init()

def printTitle(s):
    print("\n" + Fore.GREEN + Style.BRIGHT +  s + Style.RESET_ALL)

def printSubTitle(s):
    print("\n" + Style.BRIGHT + s + Style.RESET_ALL)

printTitle("Max and AbsMax")
a=np.array([1.,-3.,4.,0.,-10.,8.])
i=dsp.arm_absmax_f32(a)

printSubTitle("Fixed point tests")
# Normalize for fixed point tests
a = a / i[0]

a7 = f.toQ7(a)
a15 = f.toQ15(a)
a31 = f.toQ31(a)

print(a31)
print(dsp.arm_absmax_no_idx_q31(a31))
print(dsp.arm_max_no_idx_q31(a31))

print(a15)
print(dsp.arm_absmax_no_idx_q15(a15))
print(dsp.arm_max_no_idx_q15(a15))

print(a7)
print(dsp.arm_absmax_no_idx_q7(a7))
print(dsp.arm_max_no_idx_q7(a7))

printTitle("Min and AbsMin")
a=np.array([1.,-3.,4.,0.5,-10.,8.])
i=dsp.arm_absmax_f32(a)

printSubTitle("Fixed point tests")

# Normalize for fixed point tests
a = a / i[0]


a7 = f.toQ7(a)
a15 = f.toQ15(a)
a31 = f.toQ31(a)

print(a31)
print(dsp.arm_absmin_no_idx_q31(a31))
print(dsp.arm_min_no_idx_q31(a31))

print(a15)
print(dsp.arm_absmin_no_idx_q15(a15))
print(dsp.arm_min_no_idx_q15(a15))

print(a7)
print(dsp.arm_absmin_no_idx_q7(a7))
print(dsp.arm_min_no_idx_q7(a7))