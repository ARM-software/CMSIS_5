import cmsisdsp.fixedpoint as f
import numpy as np

#: F64 format
F64 = 64 
#: F32 format
F32 = 32
#: F16 format
F16 = 16
#: Q31 fixed point format
Q31 = 31 
#: Q15 fixed point format
Q15 = 15
#: Q7 fixed point format
Q7 = 7

class UnknownCMSISDSPDataType(Exception):
    pass

def convert(samples,format):
    """
     Return an array of scalars in a given format converted from an array of doubles.
     It is typically used to convert a reference table in double to a table with a lower
     accuracy used for a specific implementation of an algorithm.

     :param samples: array of double.
     :type samples: array
     :param format: Format identification (F64,F32,F16,Q31,Q15,Q7).
     :type format: int
     :return: array of scalars in chosen format.
     :rtype: array

     """
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