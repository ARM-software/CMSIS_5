import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation



def writeTests(config):
    NBSAMPLES=256
    NUMTAPS = 64

    samples=np.random.randn(NBSAMPLES)
    refs=np.random.randn(NBSAMPLES)
    taps=np.random.randn(NUMTAPS)

    samples = samples/max(samples)
    refs = samples/max(refs)
    taps = taps/max(taps)
    

    config.writeInput(1, samples,"Samples")
    config.writeInput(1, taps,"Coefs")
    config.writeInput(1, refs,"Refs")

    

PATTERNDIR = os.path.join("Patterns","DSP","Filtering","FIR","FIR")
PARAMDIR = os.path.join("Parameters","DSP","Filtering","FIR","FIR")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
#configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")



writeTests(configf32)
writeTests(configq31)
writeTests(configq15)
#writeTests(configq7)



