import os.path
import numpy as np
import itertools
import Tools
from scipy import signal
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,semilogx, semilogy
import math

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

def cartesian(*somelists):
   r=[]
   for element in itertools.product(*somelists):
       r.append(element)
   return(r)

def writeBenchmarks(config):
    NBSAMPLES=512 # 512 for stereo
    NUMSTAGES = 4

    samples=np.random.randn(NBSAMPLES)
    coefs=np.random.randn(NUMSTAGES*5)

    samples = Tools.normalize(samples)
    coefs = Tools.normalize(coefs)
    

    # Used for benchmarks
    config.writeInput(1, samples,"Samples")
    config.writeInput(1, coefs,"Coefs")

def getCoefs(n,sos,format):
    if format==15:
       coefs=np.reshape(np.hstack((np.insert(sos[:,:3],1,0.0,axis=1),-sos[:,4:])),n*6)
    else:
       coefs=np.reshape(np.hstack((sos[:,:3],-sos[:,4:])),n*5)
    
    if format==31:
        # Postshift must be 2 in the tests
        coefs = coefs / 4.0

    if format==15:
        # Postshift must be 2 in the tests
        coefs = coefs / 4.0

    return(coefs)

def genSos(numTaps):
    zeros=[] 
    poles=[] 
    for i in range(0,numTaps):
        phase = np.random.rand()*2.0 * math.pi
        z = np.exp(1j*phase)

        phase = np.random.rand()*2.0 * math.pi
        amplitude = np.random.rand()*0.7
        p = np.exp(1j*phase) * amplitude

        zeros += [z,np.conj(z)]
        poles += [p,np.conj(p)]
    
    g = 0.02

    sos = signal.zpk2sos(zeros,poles,g)

    return(sos)


def writeTests(config,format):
    # Write test with fixed and known patterns
    NB = 100
    t = np.linspace(0, 1,NB)

    sig = Tools.normalize(np.sin(2*np.pi*5*t)+np.random.randn(len(t)) * 0.2 + 0.4*np.sin(2*np.pi*20*t))

    if format==31:
       sig = 1.0*sig / (1 << 2)

    #if format==15:
    #   sig = 1.0*sig / 2.0

    p0 = np.exp(1j*0.05) * 0.98 
    p1 = np.exp(1j*0.25) * 0.9 
    p2 = np.exp(1j*0.45) * 0.97
    
    z0 = np.exp(1j*0.02)
    z1 = np.exp(1j*0.65)
    z2 = np.exp(1j*1.0)
    
    g = 0.02
        
    sos = signal.zpk2sos(
          [z0,np.conj(z0),z1,np.conj(z1),z2,np.conj(z2)]
         ,[p0, np.conj(p0),p1, np.conj(p1),p2, np.conj(p2)]
         ,g)

    coefs=getCoefs(3,sos,format)
    
    res=signal.sosfilt(sos,sig)

    config.writeInput(1, sig,"BiquadInput")
    config.writeInput(1, res,"BiquadOutput")
    config.writeInput(1, coefs,"BiquadCoefs")

    #if format==0:
    #   figure()
    #   plot(sig)
    #   figure()
    #   plot(res)
    #   show()

    # Now random patterns to test different tail sizes
    # and number of loops

    numStages = [Tools.loopnb(format,Tools.TAILONLY),
       Tools.loopnb(format,Tools.BODYONLY),
       Tools.loopnb(format,Tools.BODYANDTAIL)
    ]

    blockSize=[Tools.loopnb(format,Tools.TAILONLY),
       Tools.loopnb(format,Tools.BODYONLY),
       Tools.loopnb(format,Tools.BODYANDTAIL)
    ]

    allConfigs = cartesian(numStages, blockSize)
    
    allconf=[] 
    allcoefs=[]
    allsamples=[]
    allStereo=[]
    alloutputs=[]
    allStereoOutputs=[]

    for (n,b) in allConfigs:
        samples=np.random.randn(b)
        samples = Tools.normalize(samples)

        samplesB=np.random.randn(b)
        samplesB = Tools.normalize(samplesB)

        stereo = np.empty((samples.size + samplesB.size,), dtype=samples.dtype)
        stereo[0::2] = samples
        stereo[1::2] = samplesB

        sos = genSos(n)
        coefs=getCoefs(n,sos,format)
        
        output=signal.sosfilt(sos,samples)
        outputB=signal.sosfilt(sos,samplesB)

        stereoOutput = np.empty((output.size + outputB.size,), dtype=output.dtype)
        stereoOutput[0::2] = output
        stereoOutput[1::2] = outputB

        allStereoOutputs += list(stereoOutput)
        alloutputs += list(output)
        allconf += [n,b]
        allcoefs += list(coefs)
        allsamples += list(samples)
        allStereo += list(stereo)


    config.writeReferenceS16(2,allconf,"AllBiquadConfigs")
    config.writeInput(2,allsamples,"AllBiquadInputs")
    config.writeInput(2,allcoefs,"AllBiquadCoefs")
    config.writeReference(2,alloutputs,"AllBiquadRefs")
    if format==0:
        config.writeInput(2,allStereo,"AllBiquadStereoInputs")
        config.writeReference(2,allStereoOutputs,"AllBiquadStereoRefs")


    
def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Filtering","BIQUAD","BIQUAD")
    PARAMDIR = os.path.join("Parameters","DSP","Filtering","BIQUAD","BIQUAD")
    
    configf64=Tools.Config(PATTERNDIR,PARAMDIR,"f64")
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    #configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    
    
    writeBenchmarks(configf32)
    writeBenchmarks(configq31)
    writeBenchmarks(configq15)
    writeBenchmarks(configf64)

    writeTests(configf32,0)
    writeTests(configq31,31)
    writeTests(configq15,15)
    writeTests(configf64,64)
    
    #writeTests(configq7)

if __name__ == '__main__':
  generatePatterns()

