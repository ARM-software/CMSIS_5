import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

def randComplex(nb):
    data = np.random.randn(2*nb)
    data = data/max(data)
    data_comp = data.view(dtype=np.complex128)
    return(data_comp)

def asReal(a):
    #return(a.view(dtype=np.float64))
    return(a.reshape(np.size(a)).view(dtype=np.float64))

def writeTests(config):
    NBSAMPLES=256

    data1=randComplex(NBSAMPLES)
    data2=randComplex(NBSAMPLES)
    data3=np.random.randn(NBSAMPLES)
    data3 = data3/max(data3)
    

    config.writeInput(1, asReal(data1))
    config.writeInput(2, asReal(data2))
    config.writeInput(3, data3)
    
    ref = np.conj(data1)
    config.writeReference(1, asReal(ref))

    ref = np.array(np.dot(data1 ,data2))
    config.writeReference(2, asReal(ref))

    ref = np.absolute(data1)**2
    config.writeReference(3, asReal(ref))

    ref = data1 * data2
    config.writeReference(4, asReal(ref))

    ref = data1 * data3
    config.writeReference(5, asReal(ref))
    

PATTERNDIR = os.path.join("Patterns","DSP","ComplexMaths","ComplexMaths")
PARAMDIR = os.path.join("Parameters","DSP","ComplexMaths","ComplexMaths")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
#configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")



writeTests(configf32)
writeTests(configq31)
writeTests(configq15)
#writeTests(configq7)



