import numpy as np

def q31sat(x):
     if x > 0x7FFFFFFF:
          return(np.int32(0x7FFFFFFF))
     elif x < -0x80000000:
          return(np.int32(0x80000000))
     else:
          return(np.int32(x))

q31satV=np.vectorize(q31sat)

def toQ31(x):
     """
     Return an array of Q31 scalars from an array of floats

     :param x: array of float.
     :type x: array
     :return: array of Q31 scalars.
     :rtype: array

     """
     return(q31satV(np.round(np.array(x) * (1<<31))))

def q15sat(x):
     if x > 0x7FFF:
          return(np.int16(0x7FFF))
     elif x < -0x8000:
          return(np.int16(0x8000))
     else:
          return(np.int16(x))

q15satV=np.vectorize(q15sat)

def toQ15(x):
     """
     Return an array of Q15 scalars from an array of floats

     :param x: array of float.
     :type x: array
     :return: array of Q15 scalars.
     :rtype: array

     """
     return(q15satV(np.round(np.array(x) * (1<<15))))

def q7sat(x):
     if x > 0x7F:
          return(np.int8(0x7F))
     elif x < -0x80:
          return(np.int8(0x80))
     else:
          return(np.int8(x))

q7satV=np.vectorize(q7sat)

def toQ7(x):
     """
     Return an array of Q7 scalars from an array of floats

     :param x: array of float.
     :type x: array
     :return: array of Q7 scalars.
     :rtype: array

     """
     return(q7satV(np.round(np.array(x) * (1<<7))))

def Q31toF32(x):
     """
     Return an array of float values from an array of Q31 values

     :param x: Array of Q31 values.
     :type x: array
     :return: Array of float values.
     :rtype: array

     """
     return(1.0*np.array(x) / 2**31)

def Q15toF32(x):
     """
     Return an array of float values from an array of Q15 values

     :param x: Array of Q15 values.
     :type x: array
     :return: Array of float values.
     :rtype: array

     """
     return(1.0*np.array(x) / 2**15)

def Q7toF32(x):
     """
     Return an array of float values from an array of Q7 values

     :param x: Array of Q7 values.
     :type x: array
     :return: Array of float values.
     :rtype: array

     """
     return(1.0*np.array(x) / 2**7)