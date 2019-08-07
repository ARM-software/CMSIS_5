import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation



def writeTests(config):
    NBSAMPLES=512 # 512 for stereo
    NUMSTAGES = 4

    samples=np.random.randn(NBSAMPLES)
    coefs=np.random.randn(NUMSTAGES*5)

    samples = samples/max(samples)
    coefs = coefs/max(coefs)
    

    config.writeInput(1, samples,"Samples")
    config.writeInput(1, coefs,"Coefs")

    

PATTERNDIR = os.path.join("Patterns","DSP","Filtering","BIQUAD","BIQUAD")
PARAMDIR = os.path.join("Parameters","DSP","Filtering","BIQUAD","BIQUAD")

configf64=Tools.Config(PATTERNDIR,PARAMDIR,"f64")
configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
#configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")



writeTests(configf32)
writeTests(configq31)
writeTests(configq15)
writeTests(configf64)

#writeTests(configq7)



