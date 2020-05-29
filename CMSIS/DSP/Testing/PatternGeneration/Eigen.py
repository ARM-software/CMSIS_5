import os.path
import numpy as np
import itertools
import Tools


# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation


def writeTests(config,format):
    NBSAMPLES=256

    data1=np.random.randn(NBSAMPLES)
    data2=np.random.randn(NBSAMPLES)
    data3=np.random.randn(1)
    
    data1 = Tools.normalize(data1)
    data2 = Tools.normalize(data2)

    # temp for debug of f16
    config.writeInput(1, data1)
    config.writeInput(2, data2)
    
    ref = data1 + data2
    config.writeReference(1, ref)
    
    
    #nb = Tools.loopnb(format,Tools.TAILONLY)
    #nb = Tools.loopnb(format,Tools.BODYONLY)
    #nb = Tools.loopnb(format,Tools.BODYANDTAIL)
    
def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","EigenBenchmarks","VectorBenchmarks","VectorBenchmarks")
    PARAMDIR = os.path.join("Parameters","EigenBenchmarks","VectorBenchmarks","VectorBenchmarks")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    
    writeTests(configf32,0)
    
if __name__ == '__main__':
  generatePatterns()
