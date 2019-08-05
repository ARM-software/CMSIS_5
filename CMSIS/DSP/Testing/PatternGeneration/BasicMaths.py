import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

def writeTests(config):
    NBSAMPLES=256

    data1=np.random.randn(NBSAMPLES)
    data2=np.random.randn(NBSAMPLES)
    data3=np.random.randn(1)
    
    data1 = data1/max(data1)
    data2 = data1/max(data2)

    config.writeInput(1, data1)
    config.writeInput(2, data2)
    
    ref = data1 + data2
    config.writeReference(1, ref)
    
    ref = data1 - data2
    config.writeReference(2, ref)
    
    ref = data1 * data2
    config.writeReference(3, ref)
    
    ref = -data1
    config.writeReference(4, ref)
    
    ref = data1 + 0.5
    config.writeReference(5, ref)
    
    ref = data1 * 0.5
    config.writeReference(6, ref)
    
    nb = 3
    ref = np.array([np.dot(data1[0:nb] ,data2[0:nb])])
    config.writeReference(7, ref)
    
    nb = 8
    ref = np.array([np.dot(data1[0:nb] ,data2[0:nb])])
    config.writeReference(8, ref)
    
    nb = 9
    ref = np.array([np.dot(data1[0:nb] ,data2[0:nb])])
    config.writeReference(9, ref)
    
    ref = abs(data1)
    config.writeReference(10, ref)

    ref = np.array([np.dot(data1 ,data2)])
    config.writeReference(11, ref)


PATTERNDIR = os.path.join("Patterns","DSP","BasicMaths","BasicMaths")
PARAMDIR = os.path.join("Parameters","DSP","BasicMaths","BasicMaths")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")



writeTests(configf32)
writeTests(configq31)
writeTests(configq15)
writeTests(configq7)

# Params just as example
someLists=[[1,3,5],[1,3,5],[1,3,5]]

r=np.array([element for element in itertools.product(*someLists)])
configf32.writeParam(1, r.reshape(81))

