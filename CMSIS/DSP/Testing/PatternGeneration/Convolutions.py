import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation



def writeTests(config):
    NBSAMPLES=128

    inputsA=np.random.randn(NBSAMPLES)
    inputsB=np.random.randn(NBSAMPLES)

    inputsA = inputsA/max(inputsA)
    inputsB = inputsB/max(inputsB)
    

    config.writeInput(1, inputsA,"InputsA")
    config.writeInput(1, inputsB,"InputsB")

    

PATTERNDIR = os.path.join("Patterns","DSP","Filtering","MISC","MISC")
PARAMDIR = os.path.join("Parameters","DSP","Filtering","MISC","MISC")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")



writeTests(configf32)
writeTests(configq31)
writeTests(configq15)
writeTests(configq7)



