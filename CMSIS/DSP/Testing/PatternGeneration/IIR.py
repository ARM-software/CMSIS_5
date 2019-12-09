import os.path
import numpy as np
import itertools
import Tools
from scipy import signal
import math

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

#def writeBenchmarks(config):

def writeTests(config,format):
    # Write test with fixed and known patterns
    NB = Tools.loopnb(format,Tools.BODYANDTAIL)
    t = np.linspace(0, 1,NB)

    # 1st order lowpass butterworth filter
    sig = Tools.normalize(np.sin(2*np.pi*5*t)+np.random.randn(len(t)) * 0.2 + 0.4*np.sin(2*np.pi*20*t))
    config.writeInput(1, sig,"IIR_1st_Input")

    b, a = signal.butter(1, 0.05)
    coefs = [ b[0], b[1], -a[1] ] # scipy IIRs use negative a coefs
    config.writeInput(1, coefs,"IIR_1st_Coefs")

    ref = signal.lfilter(b, a, sig)
    config.writeReference(1,ref,"IIR_1st_Reference")

    b, a = signal.butter(1, 0.15)
    coefs = np.concatenate((coefs, [ b[0], b[1], -a[1] ]), axis=None)
    config.writeInput(2, coefs,"IIR_1st_Coefs")

    ref = signal.lfilter(b, a, ref)
    config.writeReference(2,ref,"IIR_1st_Reference")



    # 2nd order lowpass butterworth filter
    sig = Tools.normalize(np.sin(2*np.pi*5*t)+np.random.randn(len(t)) * 0.2 + 0.4*np.sin(2*np.pi*20*t))
    config.writeInput(1, sig,"IIR_2nd_Input")

    b, a = signal.butter(2, 0.05)
    coefs = [ b[0], b[1], b[2], -a[1], -a[2] ] # scipy IIRs use negative a coefs
    config.writeInput(1, coefs,"IIR_2nd_Coefs")

    ref = signal.lfilter(b, a, sig)
    config.writeReference(1,ref,"IIR_2nd_Reference")

    b, a = signal.butter(2, 0.15)
    coefs = np.concatenate((coefs, [ b[0], b[1], b[2], -a[1], -a[2] ]), axis=None)
    config.writeInput(2, coefs,"IIR_2nd_Coefs")

    ref = signal.lfilter(b, a, ref)
    config.writeReference(2,ref,"IIR_2nd_Reference")



    # 3rd order lowpass butterworth filter
    sig = Tools.normalize(np.sin(2*np.pi*5*t)+np.random.randn(len(t)) * 0.2 + 0.4*np.sin(2*np.pi*20*t))
    config.writeInput(1, sig,"IIR_3rd_Input")

    b, a = signal.butter(3, 0.05)
    coefs = [ b[0], b[1], b[2], b[3], -a[1], -a[2], -a[3] ] # scipy IIRs use negative a coefs
    config.writeInput(1, coefs,"IIR_3rd_Coefs")

    ref = signal.lfilter(b, a, sig)
    config.writeReference(1,ref,"IIR_3rd_Reference")

    b, a = signal.butter(3, 0.15)
    coefs = np.concatenate((coefs, [ b[0], b[1], b[2], b[3], -a[1], -a[2], -a[3] ]), axis=None)
    config.writeInput(2, coefs,"IIR_3rd_Coefs")

    ref = signal.lfilter(b, a, ref)
    config.writeReference(2,ref,"IIR_3rd_Reference")



    # 5th order lowpass butterworth filter
    sig = Tools.normalize(np.sin(2*np.pi*5*t)+np.random.randn(len(t)) * 0.2 + 0.4*np.sin(2*np.pi*20*t))
    config.writeInput(1, sig,"IIR_5th_Input")

    b, a = signal.butter(5, 0.05)
    coefs = [ b[0], b[1], b[2], b[3], b[4], b[5], -a[1], -a[2], -a[3], -a[4], -a[5] ] # scipy IIRs use negative a coefs
    config.writeInput(1, coefs,"IIR_5th_Coefs")

    ref = signal.lfilter(b, a, sig)
    config.writeReference(1,ref,"IIR_5th_Reference")

    b, a = signal.butter(5, 0.15)
    coefs = np.concatenate((coefs, [ b[0], b[1], b[2], b[3], b[4], b[5], -a[1], -a[2], -a[3], -a[4], -a[5] ]), axis=None)
    config.writeInput(2, coefs,"IIR_5th_Coefs")

    ref = signal.lfilter(b, a, ref)
    config.writeReference(2,ref,"IIR_5th_Reference")
 


def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Filtering","IIR","IIR")
    PARAMDIR = os.path.join("Parameters","DSP","Filtering","IIR","IIR")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    
    writeTests(configf32,0)

if __name__ == '__main__':
  generatePatterns()

