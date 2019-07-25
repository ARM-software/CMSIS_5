import os.path
import itertools
import Tools
import random
import numpy as np
import scipy
import scipy.stats

NBTESTS = 10
VECDIM = [12,14,20]

def entropyTest(config,nb):
    inputs = [] 
    outputs = [] 
    vecDim = VECDIM[nb % len(VECDIM)]
    dims=np.array([NBTESTS,vecDim])
    for _ in range(0,NBTESTS):
       v = np.random.rand(vecDim)
       v = v / np.sum(v)
       e = scipy.stats.entropy(v)
       inputs += list(v)
       outputs.append(e)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    config.writeInput(nb, inputs,"Input")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefEntropy")

def logsumexpTest(config,nb):
    inputs = [] 
    outputs = [] 
    vecDim = VECDIM[nb % len(VECDIM)]
    dims=np.array([NBTESTS,vecDim])
    for _ in range(0,NBTESTS):
       v = np.random.rand(vecDim)
       v = v / np.sum(v)
       e = scipy.special.logsumexp(v)
       inputs += list(v)
       outputs.append(e)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    config.writeInput(nb, inputs,"Input")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefLogSumExp")

def klTest(config,nb):
    inputsA = [] 
    inputsB = [] 
    outputs = [] 
    vecDim = VECDIM[nb % len(VECDIM)]
    dims=np.array([NBTESTS,vecDim])
    for _ in range(0,NBTESTS):
       va = np.random.rand(vecDim)
       va = va / np.sum(va)

       vb = np.random.rand(vecDim)
       vb = vb / np.sum(vb)

       e = scipy.stats.entropy(va,vb)
       inputsA += list(va)
       inputsB += list(vb)
       outputs.append(e)
    inputsA = np.array(inputsA)
    inputsB = np.array(inputsB)
    outputs = np.array(outputs)
    config.writeInput(nb, inputsA,"InputA")
    config.writeInput(nb, inputsB,"InputB")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefKL")

def logSumExpDotTest(config,nb):
    inputsA = [] 
    inputsB = [] 
    outputs = [] 
    vecDim = VECDIM[nb % len(VECDIM)]
    dims=np.array([NBTESTS,vecDim])
    for _ in range(0,NBTESTS):
       va = np.random.rand(vecDim)
       va = va / np.sum(va)

       vb = np.random.rand(vecDim)
       vb = vb / np.sum(vb)

       d = 0.001
       # It is a proba so must be in [0,1]
       # But restricted to ]d,1] so that the log exists
       va = (1-d)*va + d
       vb = (1-d)*vb + d
       e = np.log(np.dot(va,vb))
       va = np.log(va)
       vb = np.log(vb)

       inputsA += list(va)
       inputsB += list(vb)
       outputs.append(e)
    inputsA = np.array(inputsA)
    inputsB = np.array(inputsB)
    outputs = np.array(outputs)
    config.writeInput(nb, inputsA,"InputA")
    config.writeInput(nb, inputsB,"InputB")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefLogSumExpDot")

def writeTests(config):
    entropyTest(config,1)
    logsumexpTest(config,2)
    klTest(config,3)
    logSumExpDotTest(config,4)

PATTERNDIR = os.path.join("Patterns","DSP","Stats","Stats")
PARAMDIR = os.path.join("Parameters","DSP","Stats","Stats")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")

writeTests(configf32)