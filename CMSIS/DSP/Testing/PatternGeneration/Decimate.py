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
    taps=np.random.randn(NUMTAPS)

    samples = samples/max(samples)
    taps = taps/max(taps)
    

    config.writeInput(1, samples,"Samples")
    config.writeInput(1, taps,"Coefs")

    

PATTERNDIR = os.path.join("Patterns","DSP","Filtering","DECIM","DECIM")
PARAMDIR = os.path.join("Parameters","DSP","Filtering","DECIM","DECIM")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
#configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")



writeTests(configf32)
writeTests(configq31)
writeTests(configq15)
#writeTests(configq7)


# For decimation, number of samples must be a multiple of decimation factor.
# So we cannot use a generator in the test description.
numTaps = [1,2,4,8,16]
blockSizeFactor = [1,2,4,8,16]
decimationFactor = [4,5,8] 

combinations = [numTaps,blockSizeFactor,decimationFactor]
finalLength = 3 * len(numTaps) * len(decimationFactor) * len(blockSizeFactor)

r=np.array([(n,blFactor*dFactor,dFactor) for (n,blFactor,dFactor) in itertools.product(*combinations)])
r = r.reshape(finalLength)

configf32.writeParam(1, r)
configq31.writeParam(1, r)
configq15.writeParam(1, r)

# For interpolation, number taps must be a multiple of interpolation factor.
# So we cannot use a generator in the test description.
numTapsFactor = [1,2,4,8,16]
blockSize = [16,64]
interpolationFactor = [2,4,5,8,9] 

combinations = [numTapsFactor,blockSize,interpolationFactor]
finalLength = 3 * len(numTapsFactor) * len(interpolationFactor) * len(blockSize)

r=np.array([(nFactor * iFactor,bl,iFactor) for (nFactor,bl,iFactor) in itertools.product(*combinations)])
r = r.reshape(finalLength)

configf32.writeParam(2, r)
configq31.writeParam(2, r)
configq15.writeParam(2, r)



