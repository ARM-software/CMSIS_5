import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

def randComplex(nb):
    data = np.random.randn(2*nb)
    data = Tools.normalize(data)
    data_comp = data.view(dtype=np.complex128)
    return(data_comp)

def asReal(a):
    #return(a.view(dtype=np.float64))
    return(a.reshape(np.size(a)).view(dtype=np.float64))

def writeTests(config,format):
    NBSAMPLES=256

    data1=randComplex(NBSAMPLES)
    data2=randComplex(NBSAMPLES)
    data3=np.random.randn(NBSAMPLES)
    data3 = Tools.normalize(data3)
    

    config.writeInput(1, asReal(data1))
    config.writeInput(2, asReal(data2))
    config.writeInput(3, data3)
    
    ref = np.conj(data1)
    config.writeReference(1, asReal(ref))

    nb = Tools.loopnb(format,Tools.TAILONLY)
    ref = np.array(np.dot(data1[0:nb],data2[0:nb]))
    if format==31:
        ref = ref / 2**15 # Because CMSIS format is 16.48
        config.writeReferenceQ63(2, asReal(ref))
    elif format==15:
        ref = ref / 2**7 # Because CMSIS format is 8.24
        config.writeReferenceQ31(2, asReal(ref))
    else:
        config.writeReference(2, asReal(ref))

    nb = Tools.loopnb(format,Tools.BODYONLY)
    ref = np.array(np.dot(data1[0:nb] ,data2[0:nb]))
    if format==31:
        ref = ref / 2**15 # Because CMSIS format is 16.48
        config.writeReferenceQ63(3, asReal(ref))
    elif format==15:
        ref = ref / 2**7 # Because CMSIS format is 8.24
        config.writeReferenceQ31(3, asReal(ref))
    else:
        config.writeReference(3, asReal(ref))
#
    nb = Tools.loopnb(format,Tools.BODYANDTAIL)
    ref = np.array(np.dot(data1[0:nb] ,data2[0:nb]))
    if format==31:
        ref = ref / 2**15 # Because CMSIS format is 16.48
        config.writeReferenceQ63(4, asReal(ref))
    elif format==15:
        ref = ref / 2**7 # Because CMSIS format is 8.24
        config.writeReferenceQ31(4, asReal(ref))
    else:
        config.writeReference(4, asReal(ref))
#
    ref = np.absolute(data1)
    if format==31:
        ref = ref / 2 # Because CMSIS format is 2.30
    elif format==15:
        ref = ref / 2 # Because CMSIS format is 2.14
    config.writeReference(5, ref)
#
    ref = np.absolute(data1)**2
    if format==31:
        ref = ref / 4 # Because CMSIS format is 3.29
    elif format==15:
        ref = ref / 4 # Because CMSIS format is 3.13
    config.writeReference(6, ref)
#
    ref = data1 * data2
    if format==31:
        ref = ref / 4 # Because CMSIS format is 3.29
    elif format==15:
        ref = ref / 4 # Because CMSIS format is 3.13
    config.writeReference(7, asReal(ref))
#
    ref = data1 * data3
    config.writeReference(8, asReal(ref))

    ref = np.array(np.dot(data1 ,data2))
    if format==31:
        ref = ref / 2**15 # Because CMSIS format is 16.48
        config.writeReferenceQ63(9, asReal(ref))
    elif format==15:
        ref = ref / 2**7 # Because CMSIS format is 8.24
        config.writeReferenceQ31(9, asReal(ref))
    else:
        config.writeReference(9, asReal(ref))
    
def  generatePatterns():
     PATTERNDIR = os.path.join("Patterns","DSP","ComplexMaths","ComplexMaths")
     PARAMDIR = os.path.join("Parameters","DSP","ComplexMaths","ComplexMaths")
     
     configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
     configf16=Tools.Config(PATTERNDIR,PARAMDIR,"f16")
     configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
     configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
     
     
     writeTests(configf32,0)
     writeTests(configf16,16)
     writeTests(configq31,31)
     writeTests(configq15,15)

if __name__ == '__main__':
  generatePatterns()


