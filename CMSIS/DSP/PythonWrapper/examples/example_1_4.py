# New functions for version 1.4 of the Python wrapper
import cmsisdsp as dsp
import cmsisdsp.fixedpoint as f
import numpy as np
import math
import colorama
from colorama import init,Fore, Back, Style

init()

def printTitle(s):
    print("\n" + Fore.GREEN + Style.BRIGHT +  s + Style.RESET_ALL)

def printSubTitle(s):
    print("\n" + Style.BRIGHT + s + Style.RESET_ALL)

printTitle("ArcTan2")

angles=[0,
math.pi/4,
math.pi/2,
3*math.pi/4,
math.pi,
5*math.pi/4,
3*math.pi/2,
7*math.pi/4
]

x = np.cos(angles)
y = np.sin(angles)

vals=list(zip(y,x))

printSubTitle("Atan2 Referebce")
ref=np.array([np.arctan2(yv,xv) for (yv,xv) in vals])/math.pi*180
print(ref)

printSubTitle("Atan2 F32")
resf32=np.array([dsp.arm_atan2_f32(yv,xv)[1]  for (yv,xv) in vals])/math.pi*180
print(resf32)
print(np.isclose(ref,resf32,1e-6,1e-6))

printSubTitle("Atan2 Q31")
xQ31=f.toQ31(x)
yQ31=f.toQ31(y)
valsQ31=list(zip(yQ31,xQ31))

resq31=4*f.Q31toF32(np.array([dsp.arm_atan2_q31(yv,xv)[1]  for (yv,xv) in valsQ31]))/math.pi*180
print(resq31)
print(np.isclose(ref,resq31,1e-8,1e-8))

printSubTitle("Atan2 Q15")
xQ15=f.toQ15(x)
yQ15=f.toQ15(y)
valsQ15=list(zip(yQ15,xQ15))

resq15=4*f.Q15toF32(np.array([dsp.arm_atan2_q15(yv,xv)[1]  for (yv,xv) in valsQ15]))/math.pi*180
print(resq15)
print(np.isclose(ref,resq15,1e-3,1e-3))

printTitle("MSE")

NBSAMPLES = 50

def mse(a,b):
    err = a - b
    return(np.dot(err,err) / len(a))

a=np.random.randn(NBSAMPLES)
a = a / np.max(np.abs(a))

b=np.random.randn(NBSAMPLES)
b = b / np.max(np.abs(b))

printSubTitle("MSE Reference")
ref = mse(a,b)
print(ref)

printSubTitle("MSE f64")
resf64= dsp.arm_mse_f64(a,b)
print(resf64)
print(np.isclose(ref,resf64,1e-14,1e-14))

printSubTitle("MSE f32")
resf32 = dsp.arm_mse_f32(a,b)
print(resf32)
print(np.isclose(ref,resf32,1e-7,1e-7))

printSubTitle("MSE Q31")
aQ31 = f.toQ31(a)
bQ31 = f.toQ31(b)

resQ31 = f.Q31toF32(dsp.arm_mse_q31(aQ31,bQ31))
print(resQ31)
print(np.isclose(ref,resQ31,1e-7,1e-7))

aQ15 = f.toQ15(a)
bQ15 = f.toQ15(b)

printSubTitle("MSE Q15")
resQ15 = dsp.arm_mse_q15(aQ15,bQ15)
print("%04X" % resQ15)
resQ15 = f.Q15toF32(resQ15)
print(resQ15)
print(np.isclose(ref,resQ15,1e-4,1e-4))

aQ7 = f.toQ7(a)
bQ7 = f.toQ7(b)

printSubTitle("MSE Q7")
resQ7 = dsp.arm_mse_q7(aQ7,bQ7)
print("%04X" % resQ7)
resQ7 = f.Q7toF32(resQ7)
print(resQ7)
print(np.isclose(ref,resQ7,1e-2,1e-2))

