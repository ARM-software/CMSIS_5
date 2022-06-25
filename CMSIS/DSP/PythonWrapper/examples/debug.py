import cmsisdsp as dsp
import numpy as np
from scipy import signal

firf32 = dsp.arm_fir_instance_f32()
dsp.arm_fir_init_f32(firf32,3,[1.,2,3],[0,0,0,0,0,0,0])
print(firf32.numTaps())
filtered_x = signal.lfilter([3,2,1.], 1.0, [1,2,3,4,5,1,2,3,4,5])
print(filtered_x)
print(dsp.arm_fir_f32(firf32,[1,2,3,4,5]))
print(dsp.arm_fir_f32(firf32,[1,2,3,4,5]))

a=np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12]])
b=np.array([[1.,2,3,4],[5.1,6,7,8],[9.1,10,11,12]])
#print(a+b)

#print("OK")

v=dsp.arm_mat_add_f32(a,b)
print(v)
