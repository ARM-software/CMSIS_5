import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

NBA = 40
NBI = 40
NBB = 40

def randComplex(nb):
    data = np.random.randn(2*nb)
    data = data/max(data)
    data_comp = data.view(dtype=np.complex128)
    return(data_comp)

def asReal(a):
    #return(a.view(dtype=np.float64))
    return(a.reshape(np.size(a)).view(dtype=np.float64))

def writeBinaryTests(config):
    NBSAMPLESA=NBA*NBI
    NBSAMPLESB=NBI*NBB

    data1=np.random.randn(NBSAMPLESA)
    data2=np.random.randn(NBSAMPLESB)
    
    data1 = data1/max(data1)
    data2 = data1/max(data2)

    data1C=randComplex(NBSAMPLESA)
    data2C=randComplex(NBSAMPLESB)

    config.writeInput(1, data1,"InputA")
    config.writeInput(1, data2,"InputB")

    config.writeInput(1, asReal(data1C),"InputAC")
    config.writeInput(1, asReal(data2C),"InputBC")
    

def writeUnaryTests(config):
    NBSAMPLES=NBA*NBB

    data1=np.random.randn(NBSAMPLES)
    data1 = data1/max(data1)

    config.writeInput(1, data1,"InputA")

PATTERNBINDIR = os.path.join("Patterns","DSP","Matrix","Binary","Binary")
PARAMBINDIR = os.path.join("Parameters","DSP","Matrix","Binary","Binary")

configBinaryf32=Tools.Config(PATTERNBINDIR,PARAMBINDIR,"f32")
configBinaryq31=Tools.Config(PATTERNBINDIR,PARAMBINDIR,"q31")
configBinaryq15=Tools.Config(PATTERNBINDIR,PARAMBINDIR,"q15")



writeBinaryTests(configBinaryf32)
writeBinaryTests(configBinaryq31)
writeBinaryTests(configBinaryq15)

PATTERNUNDIR = os.path.join("Patterns","DSP","Matrix","Unary","Unary")
PARAMUNDIR = os.path.join("Parameters","DSP","Matrix","Unary","Unary")

configUnaryf64=Tools.Config(PATTERNUNDIR,PARAMUNDIR,"f64")
configUnaryf32=Tools.Config(PATTERNUNDIR,PARAMUNDIR,"f32")
configUnaryq31=Tools.Config(PATTERNUNDIR,PARAMUNDIR,"q31")
configUnaryq15=Tools.Config(PATTERNUNDIR,PARAMUNDIR,"q15")


writeUnaryTests(configUnaryf64)
writeUnaryTests(configUnaryf32)
writeUnaryTests(configUnaryq31)
writeUnaryTests(configUnaryq15)
