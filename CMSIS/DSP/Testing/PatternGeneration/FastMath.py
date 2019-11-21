import os.path
import numpy as np
import itertools
import Tools
import math

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

# For benchmarks
NBSAMPLES=256


def writeTests(config,format):
    
    a1=np.array([0,math.pi/4,math.pi/2,3*math.pi/4,math.pi,5*math.pi/4,3*math.pi/2,2*math.pi-1e-6])
    a2=np.array([-math.pi/4,-math.pi/2,-3*math.pi/4,-math.pi,-5*math.pi/4,-3*math.pi/2,-2*math.pi-1e-6])
    a3 = a1 + 2*math.pi  
    angles=np.concatenate((a1,a2,a3))
    refcos = np.cos(angles)
    refsin = np.sin(angles)


    vals=np.array([0.0, 0.0, 0.1,1.0,2.0,3.0,3.5,3.6])
    sqrtvals=np.sqrt(vals)

    # Negative values in CMSIS are giving 0
    vals[0] = -0.4
    sqrtvals[0] = 0.0
    
    if format != 0:
        angles=np.concatenate((a1,a2,a1))
        angles = angles / (2*math.pi)
    config.writeInput(1, angles,"Angles")
    config.writeInput(1, vals,"SqrtInput")
    config.writeReference(1, refcos,"Cos")
    config.writeReference(1, refsin,"Sin")
    config.writeReference(1, sqrtvals,"Sqrt")

    # For benchmarks
    samples=np.random.randn(NBSAMPLES)
    samples = np.abs(Tools.normalize(samples))
    config.writeInput(1, samples,"Samples")


def writeTestsF32(config,format):
    writeTests(config,format)

    data1 = np.random.randn(20)
    data1 = np.abs(data1)
    data1 = data1 + 1e-3 # To avoid zero values
    data1 = Tools.normalize(data1)

    samples=np.concatenate((np.array([0.1,0.3,0.5,1.0,2.0]) , data1))
    config.writeInput(1, samples,"LogInput")
    v = np.log(samples)
    config.writeReference(1, v,"Log")

    samples=np.concatenate((np.array([0.0,1.0]),np.linspace(-0.4,0.4)))
    config.writeInput(1, samples,"ExpInput")
    v = np.exp(samples)
    config.writeReference(1, v,"Exp")

    # For benchmarks
    samples=np.random.randn(NBSAMPLES)
    samples = np.abs(Tools.normalize(samples))
    config.writeInput(1, samples,"Samples")

    
def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","FastMath","FastMath")
    PARAMDIR = os.path.join("Parameters","DSP","FastMath","FastMath")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    
    
    writeTestsF32(configf32,0)
    writeTests(configq31,31)
    writeTests(configq15,15)


if __name__ == '__main__':
  generatePatterns()

