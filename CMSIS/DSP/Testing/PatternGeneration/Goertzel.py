import os.path
import itertools
import Tools
import random
import numpy as np
import pandas

def writeTestsF32(config, format):
    # Goertzel FFT
    # constant
    NBSAMPLES = 20
    data = np.full(NBSAMPLES, 10)
    config.writeInput(1,data,"Goertzel_Input")
    four = np.fft.fft(data)
    ref = [four.real[0], four.imag[0]] 
    config.writeReference(1,ref,"Goertzel_Reference")

    # sin
    NBSAMPLES = 16
    f0 = 1/4 # freq
    fs = 2;
    T = 1/fs
    x = np.arange(0, NBSAMPLES)*T
    #print(x)
    data = np.sin(2*np.pi*f0*x)
    #print(data)
    config.writeInput(2,data,"Goertzel_Input")
    four = np.fft.fft(data)
    #print(four.real+four.imag)
    freq = np.fft.fftfreq(x.shape[-1])*fs
    # 0, 1/8, 2/8, 3/8, ...
    #print(freq)
    ref = [four.real[2], four.imag[2]] # f=1/4 ()
    config.writeReference(2,ref,"Goertzel_Reference")

    # rand
    NBSAMPLES = 40
    data = np.random.randn(NBSAMPLES)
    config.writeInput(3,data,"Goertzel_Input")
    four = np.fft.fft(data)
    ref = [four.real[10], four.imag[10]] 
    config.writeReference(3,ref,"Goertzel_Reference")




def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Filtering","Goertzel","Goertzel")
    PARAMDIR = os.path.join("Parameters","DSP","Filtering","Goertzel","Goertzel")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
#    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
#    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
#    configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    writeTestsF32(configf32,0)


if __name__ == '__main__':
  generatePatterns()
