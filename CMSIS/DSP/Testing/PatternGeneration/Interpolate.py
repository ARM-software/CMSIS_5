import os.path
import numpy as np
import itertools
import Tools
from scipy.interpolate import interp1d

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation


def writeTests(config,format):
    NBSAMPLES=40

    x = np.linspace(0, NBSAMPLES, num=NBSAMPLES+1, endpoint=True)
    y = np.cos(-x**2/(NBSAMPLES - 1))
    f = interp1d(x, y)
    data=x+0.5
    data=data[:-1]
    z = f(data)

    if format != 0:
       data = data / 2.0**11
    if format != 0:
       config.writeInputQ31(1, data,"Input")
    else:
       config.writeInput(1, data)
    config.writeInput(1, y,"YVals")
    
    ref = z
    config.writeReference(1, ref)
    
 




def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Interpolation","Interpolation")
    PARAMDIR = os.path.join("Parameters","DSP","Interpolation","Interpolation")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    writeTests(configf32,0)
    writeTests(configq31,31)
    writeTests(configq15,15)
    writeTests(configq7,7)


if __name__ == '__main__':
  generatePatterns()
