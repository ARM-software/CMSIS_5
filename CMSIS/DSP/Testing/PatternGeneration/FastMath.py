import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation



def writeTests(config):
    NBSAMPLES=256

    samples=np.random.randn(NBSAMPLES)
    samples = np.abs(samples/max(samples))
    config.writeInput(1, samples,"Samples")

    

PATTERNDIR = os.path.join("Patterns","DSP","FastMath","FastMath")
PARAMDIR = os.path.join("Parameters","DSP","FastMath","FastMath")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")


writeTests(configf32)
writeTests(configq31)
writeTests(configq15)




