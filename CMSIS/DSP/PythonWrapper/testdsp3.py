import cmsisdsp as dsp
import numpy as np
import cmsisdsp.fixedpoint as f

# Test vlog q31 and q15
x = np.array([0.9,0.5,2**-16])

r=dsp.arm_vlog_q15(f.toQ15(x))
print(f.Q15toF32(r)*16.0)

r=dsp.arm_vlog_q31(f.toQ31(x))
print(f.Q31toF32(r)*32.0)

print(np.log(x))

print("")
# Test sin_cos 
t=20

sinRef=np.sin(t * np.pi / 180)
cosRef=np.cos(t * np.pi / 180)
print(sinRef)
print(cosRef)

s,c=dsp.arm_sin_cos_f32(t)
print(s)
print(c)

s,c=dsp.arm_sin_cos_q31(f.toQ31(t/180.0))
print(f.Q31toF32(s))
print(f.Q31toF32(c))

print("")
# Test sqrt
a=0.6
print(np.sqrt(a))

err,r=dsp.arm_sqrt_f32(a)
print(err,r)

err,r=dsp.arm_sqrt_q31(f.toQ31(a))
print(err,f.Q31toF32(r))

err,r=dsp.arm_sqrt_q15(f.toQ15(a))
print(err,f.Q15toF32(r))

err,r=dsp.arm_sqrt_f32(-a)
print(err,r)

err,r=dsp.arm_sqrt_q31(f.toQ31(-a))
print(err,f.Q31toF32(r))

err,r=dsp.arm_sqrt_q15(f.toQ15(-a))
print(err,f.Q15toF32(r))

