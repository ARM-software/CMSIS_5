import cmsisdsp.fixedpoint as f
import numpy as np

F64 = 64 
F32 = 32
F16 = 16
Q31 = 31 
Q15 = 15
Q7 = 7

class UnknownCMSISDSPDataType(Exception):
    pass

def convert(samples,format):
    if format==Q31:
       return(f.toQ31(np.array(samples)))
    if format==Q15:
       return(f.toQ15(np.array(samples)))
    if format==Q7:
       return(f.toQ7(np.array(samples)))
    if format==F64:
        return(np.array(samples).astype(dtype=np.float64))
    if format==F32:
        return(np.array(samples).astype(dtype=np.float32))
    if format==F16:
        return(np.array(samples).astype(dtype=np.float16))
    raise UnknownCMSISDSPDataType