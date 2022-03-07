import cmsisdsp as dsp 
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import entropy,tstd, tvar
from scipy.special import logsumexp
from scipy.linalg import cholesky,ldl,solve_triangular
from scipy import signal

def imToReal1D(a):
    ar=np.zeros(np.array(a.shape) * 2)
    ar[0::2]=a.real
    ar[1::2]=a.imag
    return(ar)

def realToIm1D(ar):
    return(ar[0::2] + 1j * ar[1::2])

print("Max and AbsMax")
a=np.array([1.,-3.,4.,0.,-10.,8.])

i=dsp.arm_absmax_no_idx_f32(a)
print(i)
assert i==10.0

i=dsp.arm_absmax_no_idx_f64(a)
print(i)
assert i==10.0

r,i=dsp.arm_absmax_f64(a)
assert i==4
assert r==10.0

r,i=dsp.arm_max_f64(a)
assert i==5
assert r==8.0

i=dsp.arm_max_no_idx_f32(a)
print(i)
assert i==8

i=dsp.arm_max_no_idx_f64(a)
print(i)
assert i==8

print("Min and AbsMin")
a=np.array([1.,-3.,4.,0.5,-10.,8.])

i=dsp.arm_absmin_no_idx_f32(a)
print(i)
assert i==0.5

i=dsp.arm_absmin_no_idx_f64(a)
print(i)
assert i==0.5

r,i=dsp.arm_absmin_f64(a)
assert i==3
assert r==0.5

r,i=dsp.arm_min_f64(a)
assert i==4
assert r==-10

i=dsp.arm_min_no_idx_f32(a)
print(i)
assert i==-10

i=dsp.arm_min_no_idx_f64(a)
print(i)
assert i==-10

print("Barycenter")

a=[0] * 12
w=np.array([[2] * 12])
w[0,11]=3
a[0] =[0., 0., -0.951057]
a[1] =[0., 0., 0.951057]
a[2] =[-0.850651, 0., -0.425325]
a[3] =[0.850651, 0., 0.425325]
a[4] =[0.688191, -0.5, -0.425325]
a[5] =[0.688191, 0.5, -0.425325]
a[6] =[-0.688191, -0.5, 0.425325]
a[7] =[-0.688191, 0.5, 0.425325]
a[8] =[-0.262866, -0.809017, -0.425325]
a[9] =[-0.262866, 0.809017, -0.425325]
a[10]=[0.262866, -0.809017, 0.425325]
a[11]=[0.262866, 0.809017, 0.425325]

scaled=a * w.T
ref=np.sum(scaled,axis=0)/np.sum(w)
print(ref)

result=dsp.arm_barycenter_f32(np.array(a).reshape(12*3),w.reshape(12),12,3)
print(result)

assert_allclose(ref,result,1e-6)

print("Weighted sum")

nb=10
s = np.random.randn(nb)
w = np.random.randn(nb)

ref=np.dot(s,w)/np.sum(w)
print(ref)

res=dsp.arm_weighted_sum_f32(s,w)
print(res)

assert_allclose(ref,res,2e-5)

print("Entropy")
s = np.abs(np.random.randn(nb))
s = s / np.sum(s)

ref=entropy(s)
print(ref)
res=dsp.arm_entropy_f32(s)
print(res)
assert_allclose(ref,res,1e-6)

res=dsp.arm_entropy_f64(s)
print(res)
assert_allclose(ref,res,1e-10)

print("Kullback-Leibler")
sa = np.abs(np.random.randn(nb))
sa = sa / np.sum(sa)

sb = np.abs(np.random.randn(nb))
sb = sb / np.sum(sb)

ref=entropy(sa,sb)
print(ref)
res=dsp.arm_kullback_leibler_f32(sa,sb)
print(res)
assert_allclose(ref,res,1e-6)

res=dsp.arm_kullback_leibler_f64(sa,sb)
print(res)
assert_allclose(ref,res,1e-10)

print("Logsumexp")
s = np.abs(np.random.randn(nb))
s = s / np.sum(s)

ref=logsumexp(s)
print(ref)
res=dsp.arm_logsumexp_f32(s)
print(res)
assert_allclose(ref,res,1e-6)

print("Logsumexp dot prod")
sa = np.abs(np.random.randn(nb))
sa = sa / np.sum(sa)

sb = np.abs(np.random.randn(nb))
sb = sb / np.sum(sb)

d = 0.001
# It is a proba so must be in [0,1]
# But restricted to ]d,1] so that the log exists
sa = (1-d)*sa + d
sb = (1-d)*sb + d

ref=np.log(np.dot(sa,sb))
print(ref)

sa = np.log(sa)
sb = np.log(sb)

res=dsp.arm_logsumexp_dot_prod_f32(sa,sb)
print(res)
assert_allclose(ref,res,3e-6)

print("vexp")
sa = np.random.randn(nb)

ref = np.exp(sa)
print(ref)

res=dsp.arm_vexp_f32(sa)
print(res)
assert_allclose(ref,res,1e-6)

res=dsp.arm_vexp_f64(sa)
print(res)
assert_allclose(ref,res,1e-10)


print("vlog")
sa = np.abs(np.random.randn(nb)) + 0.001

ref = np.log(sa)
print(ref)

res=dsp.arm_vlog_f32(sa)
print(res)
assert_allclose(ref,res,2e-5,1e-5)

res=dsp.arm_vlog_f64(sa)
print(res)
assert_allclose(ref,res,2e-9,1e-9)

print("Cholesky")

a=np.array([[4,12,-16],[12,37,-43],[-16,-43,98]])
ref=cholesky(a,lower=True)
print(ref)

status,res=dsp.arm_mat_cholesky_f32(a)
print(res)
assert_allclose(ref,res,1e-6,1e-6)

status,res=dsp.arm_mat_cholesky_f64(a)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("LDLT")

def swaprow(m,k,j):
    tmp = np.copy(m[j,:])
    m[j,:] = np.copy(m[k,:])
    m[k,:] = tmp 
    return(m)

# F32 test
status,resl,resd,resperm=dsp.arm_mat_ldlt_f32(a)
n=3
p=np.identity(n)
for k in range(0,n):
    p = swaprow(p,k,resperm[k])

res=resl.dot(resd).dot(resl.T)

permutedSrc=p.dot(a).dot(p.T)
print(res)
print(permutedSrc)

assert_allclose(permutedSrc,res,1e-5,1e-5)

# F64 test
print("LDLT F64")
status,resl,resd,resperm=dsp.arm_mat_ldlt_f64(a)
n=3
p=np.identity(n)
for k in range(0,n):
    p = swaprow(p,k,resperm[k])

res=resl.dot(resd).dot(resl.T)

permutedSrc=p.dot(a).dot(p.T)
print(res)
print(permutedSrc)

assert_allclose(permutedSrc,res,1e-9,1e-9)

print("Solve lower triangular")
a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
b = np.array([[4,2,4,2],[8,4,8,4]]).T
x = solve_triangular(a, b,lower=True)
print(a)
print(b)
print(x)

b = np.array([[4,2,4,2],[8,4,8,4]]).T
status,res=dsp.arm_mat_solve_lower_triangular_f32(a,b)
print(res)
assert_allclose(x,res,1e-5,1e-5)


b = np.array([[4,2,4,2],[8,4,8,4]]).T
status,res=dsp.arm_mat_solve_lower_triangular_f64(a,b)
print(res)
assert_allclose(x,res,1e-9,1e-9)


print("Solve upper triangular")
a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
b = np.array([[4,2,4,2],[8,4,8,4]]).T
x = solve_triangular(a.T, b,lower=False)
print(a.T)
print(b)
print(x)

b = np.array([[4,2,4,2],[8,4,8,4]]).T
status,res=dsp.arm_mat_solve_upper_triangular_f32(a.T,b)
print(res)
assert_allclose(x,res,1e-5,1e-5)


b = np.array([[4,2,4,2],[8,4,8,4]]).T
status,res=dsp.arm_mat_solve_upper_triangular_f64(a.T,b)
print(res)
assert_allclose(x,res,1e-9,1e-9)


print("Mat mult f64")
a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
b = np.array([[4,2,4,2],[8,4,8,4]]).T 

ref =a.dot(b)
print(ref)

status,res = dsp.arm_mat_mult_f64(a,b)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("mat sub f64")
a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
b = a.T 

ref = a - b
print(ref)

status,res = dsp.arm_mat_sub_f64(a,b)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("abs f64")
s = np.random.randn(nb)
ref = np.abs(s)
res=dsp.arm_abs_f64(s)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("add f64")
sa = np.random.randn(nb)
sb = np.random.randn(nb)
ref = sa + sb
res=dsp.arm_add_f64(sa,sb)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("sub f64")
sa = np.random.randn(nb)
sb = np.random.randn(nb)
ref = sa - sb
res=dsp.arm_sub_f64(sa,sb)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("dot prod  f64")
sa = np.random.randn(nb)
sb = np.random.randn(nb)
ref = sa.dot(sb)
res=dsp.arm_dot_prod_f64(sa,sb)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("mult f64")
sa = np.random.randn(nb)
sb = np.random.randn(nb)
ref = sa * sb
res=dsp.arm_mult_f64(sa,sb)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("negate f64")
sa = np.random.randn(nb)
ref = -sa
res=dsp.arm_negate_f64(sa)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("offset f64")
sa = np.random.randn(nb)
ref = sa + 0.1
res=dsp.arm_offset_f64(sa,0.1)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("scale f64")
sa = np.random.randn(nb)
ref = sa * 0.1
res=dsp.arm_scale_f64(sa,0.1)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("mean f64")
sa = np.random.randn(nb)
ref = np.mean(sa)
res=dsp.arm_mean_f64(sa)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("power f64")
sa = np.random.randn(nb)
ref = np.sum(sa * sa)
res=dsp.arm_power_f64(sa)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("std f64")
sa = np.random.randn(nb)
ref = tstd(sa)
res=dsp.arm_std_f64(sa)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("variance f64")
sa = np.random.randn(nb)
ref = tvar(sa)
res=dsp.arm_var_f64(sa)
print(ref)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("fill f64")
nb=20
ref = np.ones(nb)*4.0 
res = dsp.arm_fill_f64(4.0,nb)
assert_allclose(ref,res,1e-10,1e-10)

print("copy f64")
nb=20
sa = np.random.randn(nb)
ref = sa
res = dsp.arm_copy_f64(sa)
assert_allclose(ref,res,1e-10,1e-10)

print("arm_div_q63_to_q31")
den=0x7FFF00000000 
num=0x10000
ref=den//num

res=dsp.arm_div_q63_to_q31(den,num)
print(ref)
print(res)

print("fir f64")

firf64 = dsp.arm_fir_instance_f64()
dsp.arm_fir_init_f64(firf64,3,[1.,2,3],[0,0,0,0,0,0,0])
filtered_x = signal.lfilter([3,2,1.], 1.0, [1,2,3,4,5,1,2,3,4,5])
print(filtered_x)
ra=dsp.arm_fir_f64(firf64,[1,2,3,4,5])
rb=dsp.arm_fir_f64(firf64,[1,2,3,4,5])
assert ((filtered_x == np.hstack([ra,rb])).all)

print("arm_cmplx_mag")
sa = np.random.randn(nb)
ca = realToIm1D(sa)
ref = np.abs(ca)
print(ref)
res=dsp.arm_cmplx_mag_f32(sa)
print(res)
assert_allclose(ref,res,1e-6,1e-6)
res=dsp.arm_cmplx_mag_f64(sa)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("arm_cmplx_mag_squared")
sa = np.random.randn(nb)
ca = realToIm1D(sa)
ref = np.abs(ca) * np.abs(ca)
print(ref)
res=dsp.arm_cmplx_mag_squared_f32(sa)
print(res)
assert_allclose(ref,res,1e-6,1e-6)
res=dsp.arm_cmplx_mag_squared_f64(sa)
print(res)
assert_allclose(ref,res,1e-10,1e-10)

print("cmplx mult")
sa = np.random.randn(nb)
ca = realToIm1D(sa)
sb = np.random.randn(nb)
cb = realToIm1D(sb)
ref = imToReal1D(ca * cb)
print(ref)
res = dsp.arm_cmplx_mult_cmplx_f32(sa,sb)
print(res)
assert_allclose(ref,res,1e-6,1e-6)

res = dsp.arm_cmplx_mult_cmplx_f64(sa,sb)
print(res)
assert_allclose(ref,res,1e-10,1e-10)
