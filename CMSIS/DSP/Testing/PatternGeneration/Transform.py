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
    NBRSAMPLES=2048
    NBCSAMPLES=256

    samples=np.random.randn(NBRSAMPLES)
    samples = np.abs(samples/max(samples))

    samplesC=randComplex(NBCSAMPLES)

    config.writeInput(1, samples,"RealSamples")
    config.writeInput(1, asReal(samplesC),"ComplexSamples")


PATTERNDIR = os.path.join("Patterns","DSP","Transform","Transform")
PARAMDIR = os.path.join("Parameters","DSP","Transform","Transform")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")


writeTests(configf32)
writeTests(configq31)
writeTests(configq15)




