import os.path
import numpy as np
import itertools
import Tools
from scipy.signal import firwin
import scipy.signal 
import math
from scipy.signal import upfirdn

#from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,semilogx, semilogy



# scipy 1.4.0 at lest is needed

# Those patterns are used for tests and benchmarks.
# This is containing patterns for decimation and interpolation

def cartesian(*somelists):
   r=[]
   for element in itertools.product(*somelists):
       r.append(element)
   return(r)

def writeBenchmarks(config):
    NBSAMPLES=256
    NUMTAPS = 64

    samples=np.random.randn(NBSAMPLES)
    taps=np.random.randn(NUMTAPS)

    samples = Tools.normalize(samples)
    taps =Tools.normalize(taps)
    

    config.writeInput(1, samples,"Samples")
    config.writeInput(1, taps,"Coefs")

    
def generateBenchmarkPatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Filtering","DECIM","DECIM")
    PARAMDIR = os.path.join("Parameters","DSP","Filtering","DECIM","DECIM")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    #configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    
    
    writeBenchmarks(configf32)
    writeBenchmarks(configq31)
    writeBenchmarks(configq15)
    #writeBenchmarks(configq7)
    
    
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
    numTapsFactor = [1,2,4,8]
    blockSize = [16,64]
    interpolationFactor = [2,4,5,8,9] 
    
    combinations = [numTapsFactor,blockSize,interpolationFactor]
    finalLength = 3 * len(numTapsFactor) * len(interpolationFactor) * len(blockSize)
    
    r=np.array([(nFactor * iFactor,bl,iFactor) for (nFactor,bl,iFactor) in itertools.product(*combinations)])
    r = r.reshape(finalLength)
    
    configf32.writeParam(2, r)
    configq31.writeParam(2, r)
    configq15.writeParam(2, r)

def writeDecimateTests(config,startNb,format):
    decimates=[1,2,4,8]
    blocksFactor=[Tools.loopnb(format,Tools.TAILONLY),
       Tools.loopnb(format,Tools.BODYONLY),
       Tools.loopnb(format,Tools.BODYANDTAIL)
       ]

    numTaps =[
       Tools.loopnb(format,Tools.TAILONLY),
       Tools.loopnb(format,Tools.BODYONLY),
       Tools.loopnb(format,Tools.BODYANDTAIL),
       Tools.loopnb(format,Tools.TAILONLY)+1,
       Tools.loopnb(format,Tools.BODYONLY)+1,
       Tools.loopnb(format,Tools.BODYANDTAIL)+1
    ]

    #decimates=[2]
    #blocks=[1]
    #numTaps =[8,16]
    

    #if format==15:
    #    factor=5
    #else:
    #    factor=6
    factor = 1
    ref = []
   
    allConfigs=cartesian(decimates,blocksFactor,numTaps)

    allsamples=[]
    allcoefs=[]
    alloutput=[]

    for (q,blockF,numTaps) in allConfigs:
        b = np.array(list(range(1,numTaps+1)))/(3.0*(numTaps+2))

        # nbsamples must be multiple of q
        nbsamples=factor*blockF*q
        samples=np.random.randn(nbsamples)
        samples=Tools.normalize(samples)

        
        #output=scipy.signal.decimate(samples,q,ftype=scipy.signal.dlti(b,1.0),zero_phase=False)
        output=upfirdn(b,samples,up=1,down=q,axis=-1,mode='constant',cval=0)
        output=output[0:factor*blockF]
        #print(debug-output)

        allsamples += list(samples)
        alloutput += list(output)
        allcoefs += list(reversed(b))


        
        ref += [q,len(b),len(samples),len(output)]


    config.writeInput(startNb, allsamples)
    config.writeInput(startNb, allcoefs,"Coefs")
    config.writeReference(startNb, alloutput)

    config.writeInputU32(2, ref,"Configs")

    startNb = startNb + 1

    return(startNb)


def writeInterpolateTests(config,startNb,format):
    #interpolate=[2,4,8]
    #blocks=[1,2,4,8,16]
    #numTaps =[1,2,4,8,16]

    interpolate=[1,2,4,5,8,9]
    blocks=[Tools.loopnb(format,Tools.TAILONLY),
       Tools.loopnb(format,Tools.BODYONLY),
       Tools.loopnb(format,Tools.BODYANDTAIL),]
    numTapsFactor =[4,8]


    ref = []
   
    allConfigs=cartesian(interpolate,blocks,numTapsFactor)

    allsamples=[]
    allcoefs=[]
    alloutput=[]

    for (q,blockSize,numTapsF) in allConfigs:
        # numTaps must be a multiple of q
        numTaps = numTapsF * q
        b = np.array(list(range(1,numTaps+1)))/(3.0*(numTaps+2))

        nbsamples=blockSize
        samples=np.random.randn(nbsamples)
        samples=Tools.normalize(samples)

        
        #output=scipy.signal.decimate(samples,q,ftype=scipy.signal.dlti(b,1.0),zero_phase=False)
        output=upfirdn(b,samples,up=q,down=1,axis=-1,mode='constant',cval=0)
        output=output[0:blockSize*q]
        #print(debug-output)

        allsamples += list(samples)
        alloutput += list(output)
        allcoefs += list(reversed(b))


        
        ref += [q,len(b),len(samples),len(output)]


    config.writeInput(startNb, allsamples)
    config.writeInput(startNb, allcoefs,"Coefs")
    config.writeReference(startNb, alloutput)

    config.writeInputU32(startNb, ref,"Configs")

    startNb = startNb + 1

    return(startNb)

def writeTests(config,format):
    # Benchmark ID is finishing at 1
    # So we start at 2 for file ID for tests.
    startNb = 2

    startNb=writeDecimateTests(config,startNb,format)
    startNb=writeInterpolateTests(config,startNb,format)

    

def generateTestPatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Filtering","DECIM","DECIM")
    PARAMDIR = os.path.join("Parameters","DSP","Filtering","DECIM","DECIM")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")

    writeTests(configf32,0)
    writeTests(configq31,31)
    writeTests(configq15,15)

if __name__ == '__main__':
  generateBenchmarkPatterns()
  generateTestPatterns()


