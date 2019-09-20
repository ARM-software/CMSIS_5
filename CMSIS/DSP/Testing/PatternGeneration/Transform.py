import os.path
import numpy as np
import itertools
import Tools
import scipy.fftpack

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation


FFTSIZES=[16,32,64,128,256,512,1024,2048,4096]
SINES=[0.25,0.5,0.9]
NOISES=[0.1,0.4]

def randComplex(nb):
    data = np.random.randn(2*nb)
    data = data/max(data)
    data_comp = data.view(dtype=np.complex128)
    return(data_comp)

def asReal(a):
    #return(a.view(dtype=np.float64))
    return(a.reshape(np.size(a)).view(dtype=np.float64))

def randComplex(nb):
    data = np.random.randn(2*nb)
    data = data/max(data)
    data_comp = data.view(dtype=np.complex128)
    return(data_comp)

def asReal(a):
    #return(a.view(dtype=np.float64))
    return(a.reshape(np.size(a)).view(dtype=np.float64))

def noiseSignal(nb):
    return(np.random.randn(nb))

def sineSignal(freqRatio,nb):
    fc = nb / 2.0
    f = freqRatio*fc 
    time = np.arange(0,nb)
    return(np.sin(2 * np.pi * f *  time/nb))

def noisySineSignal(noiseAmp,r,nb):
    return(noiseAmp*noiseSignal(nb) + r*sineSignal(0.25,nb))

def stepSignal(r,nb):
   n = int(nb/2)
   return(np.concatenate((np.zeros(n), r*np.ones(n))))

def writeFFTForSignal(config,sig,scaling,i,j,nb,signame):
    fft=scipy.fftpack.fft(sig)
    ifft = np.copy(fft)
    if scaling:
        fft = np.array([x/2**scaling[j] for x in fft])
    config.writeInput(i, asReal(sig),"ComplexInputSamples_%s_%d_" % (signame,nb))
    config.writeInput(i, asReal(fft),"ComplexFFTSamples_%s_%d_" % (signame,nb))
    config.writeInput(i, asReal(fft),"ComplexInputIFFTSamples_%s_%d_" % (signame,nb))

def writeTests(configs):
    i = 1

    # Write FFT tests for sinusoid
    j = 0
    for nb in FFTSIZES:
        sig = noisySineSignal(0.05,0.7,nb)
        sig = np.array([complex(x) for x in sig])
        for config,scaling in configs:
            writeFFTForSignal(config,sig,scaling,i,j,nb,"Noisy")
        i = i + 1
        j = j + 1

    # Write FFT tests for step
    j = 0
    for nb in FFTSIZES:
        sig = stepSignal(0.9,nb)
        sig = np.array([complex(x) for x in sig])
        for config,scaling in configs:
            writeFFTForSignal(config,sig,scaling,i,j,nb,"Step")
        i = i + 1
        j = j + 1

    data1=np.random.randn(512)
    data1 = data1/max(data1)
    for config,scaling in configs:
        config.writeInput(i, data1,"RealInputSamples" )

   

PATTERNDIR = os.path.join("Patterns","DSP","Transform","Transform")
PARAMDIR = os.path.join("Parameters","DSP","Transform","Transform")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")


scalings = [4,5,6,7,8,9,10,11,12]
writeTests([(configf32,None)
    ,(configq31,scalings)
    ,(configq15,scalings)])



