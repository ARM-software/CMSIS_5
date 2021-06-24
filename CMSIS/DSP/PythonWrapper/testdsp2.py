import cmsisdsp as dsp
import numpy as np
from scipy import signal
from scipy.fftpack import dct 
import fixedpoint as f
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


def imToReal2D(a):
    ar=np.zeros(np.array(a.shape) * [1,2])
    ar[::,0::2]=a.real
    ar[::,1::2]=a.imag
    return(ar)

def realToIm2D(ar):
    return(ar[::,0::2] + 1j * ar[::,1::2])

def normalize(a):
  return(a/np.max(np.abs(a)))

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

#################### MAX AND ABSMAX ##################################
printTitle("Max and AbsMax")
a=np.array([1.,-3.,4.,0.,-10.,8.])

printSubTitle("Float tests")
i=dsp.arm_max_f32(a)
print(i)

i=dsp.arm_absmax_f32(a)
print(i)

printSubTitle("Fixed point tests")

# Normalize for fixed point tests
a = a / i[0]

a31 = f.toQ31(a)
i=dsp.arm_absmax_q31(a31)
print(f.Q31toF32(i[0]),i[1])

a8 = f.toQ15(a)
i=dsp.arm_absmax_q15(a8)
print(f.Q15toF32(i[0]),i[1])

a7 = f.toQ7(a)
i=dsp.arm_absmax_q7(a7)
print(f.Q7toF32(i[0]),i[1])

################### MIN AND ABSMIN ################################

printTitle("Min and AbsMin")
a=np.array([1.,-3.,4.,0.5,-10.,8.])

printSubTitle("Float tests")
i=dsp.arm_min_f32(a)
print(i)

i=dsp.arm_absmin_f32(a)
print(i)

printSubTitle("Fixed point tests")

# Normalize for fixed point tests
idx=i[1]
i=dsp.arm_absmax_f32(a)
a = a / i[0]
print(a)
print(a[idx])

a31 = f.toQ31(a)
i=dsp.arm_absmin_q31(a31)
print(f.Q31toF32(i[0]),i[1])

a8 = f.toQ15(a)
i=dsp.arm_absmin_q15(a8)
print(f.Q15toF32(i[0]),i[1])

a7 = f.toQ7(a)
i=dsp.arm_absmin_q7(a7)
print(f.Q7toF32(i[0]),i[1])

##################### CLIPPING ###################
printTitle("Clipping tests tests")
a=np.array([1.,-3.,4.,0.5,-10.,8.])
i=dsp.arm_absmax_f32(a)

minBound =-5.0 
maxBound =6.0
b=dsp.arm_clip_f32(a,minBound,maxBound)
print(a)
print(b)

a = a / i[0]
print(a)
minBound = minBound / i[0]
maxBound = maxBound / i[0]
print(minBound,maxBound)

b=dsp.arm_clip_q31(f.toQ31(a),f.toQ31(minBound),f.toQ31(maxBound))
print(f.Q31toF32(b))

b=dsp.arm_clip_q15(f.toQ15(a),f.toQ15(minBound),f.toQ15(maxBound))
print(f.Q15toF32(b))

b=dsp.arm_clip_q7(f.toQ7(a),f.toQ7(minBound),f.toQ7(maxBound))
print(f.Q7toF32(b))

############### MAT VECTOR MULT

printTitle("Matrix x Vector")
a=np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12]])
b=np.array([-2,-1,3,4])

c = np.dot(a,b)
print(c)
c = dsp.arm_mat_vec_mult_f32(a,b)
print(c)

printSubTitle("Fixed point")
normalizationFactor=2.0*np.sqrt(np.max(np.abs(c)))
a=a/normalizationFactor
b=b/normalizationFactor
print(np.dot(a,b))

c=dsp.arm_mat_vec_mult_q31(f.toQ31(a),f.toQ31(b))
print(f.Q31toF32(c))

c=dsp.arm_mat_vec_mult_q15(f.toQ15(a),f.toQ15(b))
print(f.Q15toF32(c))

c=dsp.arm_mat_vec_mult_q7(f.toQ7(a),f.toQ7(b))
print(f.Q7toF32(c))

############### MATRIX MULTIPLY

printTitle("Matrix x Matrix")

a=np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12]])
b=np.array([[1.,2,3],[5.1,6,7],[9.1,10,11],[5,8,4]])
print(np.dot(a , b))
c=dsp.arm_mat_mult_f32(a,b)
print(c[1])

printSubTitle("Fixed point")

normalizationFactor=2.0*np.sqrt(np.max(np.abs(c[1])))
a = a / normalizationFactor
b = b / normalizationFactor
c=dsp.arm_mat_mult_f32(a,b)
print(c[1])

print("")
af = f.toQ31(a)
bf = f.toQ31(b)
c = dsp.arm_mat_mult_q31(af,bf)
print(f.Q31toF32(c[1]))

print("")
af = f.toQ15(a)
bf = f.toQ15(b)
s=bf.shape 
nb=s[0]*s[1]
tmp=np.zeros(nb)
c = dsp.arm_mat_mult_q15(af,bf,tmp)
print(f.Q15toF32(c[1]))

print("")
af = f.toQ7(a)
bf = f.toQ7(b)
s=bf.shape 
nb=s[0]*s[1]
tmp=np.zeros(nb)
c = dsp.arm_mat_mult_q7(af,bf,tmp)
print(f.Q7toF32(c[1]))

################# MAT TRANSPOSE #################

printTitle("Transposition")
a=np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12]])
normalizationFactor=np.max(np.abs(c[1]))
a = a / normalizationFactor

print(np.transpose(a))
print("")
r=dsp.arm_mat_trans_f32(a)
print(r[1])
print("")

r=dsp.arm_mat_trans_q31(f.toQ31(a))
print(f.Q31toF32(r[1]))
print("")

r=dsp.arm_mat_trans_q15(f.toQ15(a))
print(f.Q15toF32(r[1]))
print("")

r=dsp.arm_mat_trans_q7(f.toQ7(a))
print(f.Q7toF32(r[1]))
print("")

################## FILL FUNCTIONS #################

v=0.22 
nb=10 
a=np.full((nb,),v)
print(a)

a=dsp.arm_fill_f32(v,nb)
print(a)

a=f.Q31toF32(dsp.arm_fill_q31(f.toQ31(v),nb))
print(a)

a=f.Q15toF32(dsp.arm_fill_q15(f.toQ15(v),nb))
print(a)

a=f.Q7toF32(dsp.arm_fill_q7(f.toQ7(v),nb))
print(a)

################# COMPLEX MAT TRANSPOSE #################

printTitle("Complex Transposition")
a=np.array([[1. + 0.0j ,2 + 1.0j,3 + 0.0j,4 + 2.0j],
            [5 + 1.0j,6 + 2.0j,7 + 3.0j,8 + 1.0j],
            [9 - 2.0j,10 + 1.0j,11 - 4.0j,12 + 1.0j]])
normalizationFactor=np.max(np.abs(c[1]))
a = a / normalizationFactor

print(np.transpose(a))
print("")
r=dsp.arm_mat_cmplx_trans_f32(imToReal2D(a))
print(realToIm2D(r[1]))
print("")

r=dsp.arm_mat_cmplx_trans_q31(f.toQ31(imToReal2D(a)))
print(realToIm2D(f.Q31toF32(r[1])))
print("")

r=dsp.arm_mat_cmplx_trans_q15(f.toQ15(imToReal2D(a)))
print(realToIm2D(f.Q15toF32(r[1])))
print("")

################ Levinson ##################

printTitle("Levinson Durbin")
na=5
s = np.random.randn(na+1)
s = normalize(s)
phi = autocorr(s)
phi = normalize(phi)

sigmav,arcoef,pacf,sigma,phi1=statsmodels.tsa.stattools.levinson_durbin(phi,nlags=na,isacov=True)
      
print(arcoef)
print(sigmav)

(a,err)=dsp.arm_levinson_durbin_f32(phi,na)
print(a)
print(err)

phiQ31 = f.toQ31(phi)
(aQ31,errQ31)=dsp.arm_levinson_durbin_q31(phiQ31,na)
print(f.Q31toF32(aQ31))
print(f.Q31toF32(errQ31))

################## Bitwise operations #################

printTitle("Bitwise operations")
def genBitvectors(nb,format):
    if format == 31:
       maxVal = 0x7fffffff
    if format == 15:
       maxVal = 0x7fff
    if format == 7:
       maxVal = 0x7f 

    minVal = -maxVal-1
    
    return(np.random.randint(minVal, maxVal, size=nb))

NBSAMPLES=10



printSubTitle("u32")
su32A=genBitvectors(NBSAMPLES,31)
su32B=genBitvectors(NBSAMPLES,31)
ffff = (np.ones(NBSAMPLES)*(-1)).astype(np.int)


ref=np.bitwise_and(su32A, su32B)
#print(ref)
result=dsp.arm_and_u32(su32A, su32B).astype(int)
print(result-ref)

ref=np.bitwise_or(su32A, su32B)
#print(ref)
result=dsp.arm_or_u32(su32A, su32B).astype(int)
print(result-ref)

ref=np.bitwise_xor(su32A, su32B)
#print(ref)
result=dsp.arm_xor_u32(su32A, su32B).astype(int)
print(result-ref)

ref=np.bitwise_xor(ffff, su32A)
#print(ref)
result=dsp.arm_not_u32(su32A).astype(int)
print(result-ref)

printSubTitle("u16")
su16A=genBitvectors(NBSAMPLES,15)
su16B=genBitvectors(NBSAMPLES,15)

ffff = (np.ones(NBSAMPLES)*(-1)).astype(np.int)


ref=np.bitwise_and(su16A, su16B)
#print(ref)
result=dsp.arm_and_u16(su16A, su16B).astype(np.short)
print(result-ref)

ref=np.bitwise_or(su16A, su16B)
#print(ref)
result=dsp.arm_or_u16(su16A, su16B).astype(np.short)
print(result-ref)

ref=np.bitwise_xor(su16A, su16B)
#print(ref)
result=dsp.arm_xor_u16(su16A, su16B).astype(np.short)
print(result-ref)

ref=np.bitwise_xor(ffff, su16A)
#print(ref)
result=dsp.arm_not_u16(su16A).astype(np.short)
print(result-ref)

printSubTitle("u8")

su8A=genBitvectors(NBSAMPLES,7)
su8B=genBitvectors(NBSAMPLES,7)

ref=np.bitwise_and(su8A, su8B)
#print(ref)
result=dsp.arm_and_u8(su8A, su8B).astype(np.byte)
print(result-ref)

ref=np.bitwise_or(su8A, su8B)
#print(ref)
result=dsp.arm_or_u8(su8A, su8B).astype(np.byte)
print(result-ref)

ref=np.bitwise_xor(su8A, su8B)
#print(ref)
result=dsp.arm_xor_u8(su8A, su8B).astype(np.byte)
print(result-ref)

ref=np.bitwise_xor(ffff, su8A)
#print(ref)
result=dsp.arm_not_u8(su8A).astype(np.byte)
print(result-ref)

#################### Quaternion tests ##################
NBSAMPLES=3

def flattenQuat(l):
    return(np.array([list(x) for x in l]).reshape(4*len(l)))

def flattenRot(l):
    return(np.array([list(x) for x in l]).reshape(9*len(l)))

# q and -q are representing the same rotation.
# So there is an ambiguity for the tests.
# We force the real part of be positive.
def mkQuaternion(mat):
    q=Quaternion(matrix=mat)
    if q.scalar < 0:
        return(-q)
    else:
        return(q)

a=[2.0*Quaternion.random() for x in range(NBSAMPLES)]
src=flattenQuat(a)


res=flattenQuat([x.normalised for x in a])
print(res)
output=dsp.arm_quaternion_normalize_f32(src)
print(output)
print("")

res=flattenQuat([x.conjugate for x in a])
print(res)
output=dsp.arm_quaternion_conjugate_f32(src)
print(output)
print("")

res=flattenQuat([x.inverse for x in a])
print(res)
output=dsp.arm_quaternion_inverse_f32(src)
print(output)
print("")

res=[x.norm for x in a]
print(res)
output=dsp.arm_quaternion_norm_f32(src)
print(output)
print("")

a=[x.normalised for x in a]
ra=[x.rotation_matrix for x in a]
rb=[mkQuaternion(x) for x in ra]

srca=flattenQuat(a)
resa=dsp.arm_quaternion2rotation_f32(srca)
resb=dsp.arm_rotation2quaternion_f32(resa)


print(ra)
print(resa)
print("")
print(rb)
print(resb)#

a=[2.0*Quaternion.random() for x in range(NBSAMPLES)]
b=[2.0*Quaternion.random() for x in range(NBSAMPLES)]

c = np.array(a) * np.array(b)
print(c)

srca=flattenQuat(a)
srcb=flattenQuat(b)
resc=dsp.arm_quaternion_product_f32(srca,srcb)

print(resc)

print(a[0]*b[0])
res=dsp.arm_quaternion_product_single_f32(srca[0:4],srcb[0:4])
print(res)

