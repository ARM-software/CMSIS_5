import os.path
import itertools
import Tools
import random
import numpy as np
import scipy
import scipy.stats
import math

NBTESTS = 10
VECDIM = [12,14,20]

def entropyTest(config,nb):
    DIMS = [3,8,9,12]
    inputs = [] 
    outputs = [] 
    dims=[NBTESTS]
    for i in range(0,NBTESTS):
       vecDim = DIMS[i % len(DIMS)]
       dims.append(vecDim)
       v = np.random.rand(vecDim)
       v = v / np.sum(v)
       e = scipy.stats.entropy(v)
       inputs += list(v)
       outputs.append(e)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    dims = np.array(dims)
    config.writeInput(nb, inputs,"Input")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefEntropy")

def logsumexpTest(config,nb):
    DIMS = [3,8,9,12]
    inputs = [] 
    outputs = [] 
    dims=[NBTESTS]
    for i in range(0,NBTESTS):
       vecDim = DIMS[i % len(DIMS)]
       dims.append(vecDim)
       v = np.random.rand(vecDim)
       v = v / np.sum(v)
       e = scipy.special.logsumexp(v)
       inputs += list(v)
       outputs.append(e)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    dims = np.array(dims)
    config.writeInput(nb, inputs,"Input")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefLogSumExp")

def klTest(config,nb):
    DIMS = [3,8,9,12]
    inputsA = [] 
    inputsB = [] 
    outputs = [] 
    vecDim = VECDIM[nb % len(VECDIM)]
    dims=[NBTESTS]
    for i in range(0,NBTESTS):
       vecDim = DIMS[i % len(DIMS)]
       dims.append(vecDim)
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
    dims = np.array(dims)
    config.writeInput(nb, inputsA,"InputA")
    config.writeInput(nb, inputsB,"InputB")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefKL")

def logSumExpDotTest(config,nb):
    DIMS = [3,8,9,12]
    inputsA = [] 
    inputsB = [] 
    outputs = [] 
    vecDim = VECDIM[nb % len(VECDIM)]
    dims=[NBTESTS]
    for i in range(0,NBTESTS):
       vecDim = DIMS[i % len(DIMS)]
       dims.append(vecDim)
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
    dims = np.array(dims)
    config.writeInput(nb, inputsA,"InputA")
    config.writeInput(nb, inputsB,"InputB")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, outputs,"RefLogSumExpDot")

def writeF16OnlyTests(config,nb):
    entropyTest(config,nb)
    logsumexpTest(config,nb+1)
    klTest(config,nb+2)
    logSumExpDotTest(config,nb+3)
    return(nb+4)

def writeF32OnlyTests(config,nb):
    entropyTest(config,nb)
    logsumexpTest(config,nb+1)
    klTest(config,nb+2)
    logSumExpDotTest(config,nb+3)
    return(nb+4)

def writeF64OnlyTests(config,nb):
    entropyTest(config,nb)
    klTest(config,nb+2)
    return(nb+4)

# For index in min and max we need to ensure that the difference between values
# of the input is big enough to be representable on q31, q15 or q7.
# Otherwise python will compute an index different from the one
# computed by CMSIS which is normal but then the CMSIS test will fail.

#vfunc = np.vectorize(squarer)

def floatRound(x,f):
    return(np.round(x * 2**f)/2**f)

# Min / Max tests
def generateMaxTests(config,nb,format,data):

    
    indexes=[]
    maxvals=[]

    nbiters = Tools.loopnb(format,Tools.TAILONLY)
    index=np.argmax(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYONLY)
    index=np.argmax(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYANDTAIL)
    index=np.argmax(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    if format == 7:
      # Force max at position 280
  
      nbiters = 280
  
      data = np.zeros(nbiters)
  
      data[nbiters-1] = 0.9 
      data[nbiters-2] = 0.8 
  
      index=np.argmax(data[0:nbiters])
      maxvalue=data[index]
  
      indexes.append(index)
      maxvals.append(maxvalue)

      config.writeInput(nb, data,"InputMaxIndexMax")

    config.writeReference(nb, maxvals,"MaxVals")
    config.writeInputS16(nb, indexes,"MaxIndexes")
    return(nb+1)

def generateMinTests(config,nb,format,data):

    
    indexes=[]
    maxvals=[]

    nbiters = Tools.loopnb(format,Tools.TAILONLY)
    index=np.argmin(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYONLY)
    index=np.argmin(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYANDTAIL)
    index=np.argmin(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    if format == 7:
       # Force max at position 280
       nbiters = 280
   
       data = 0.9*np.ones(nbiters)
   
       data[nbiters-1] = 0.0 
       data[nbiters-2] = 0.1 
   
       index=np.argmin(data[0:nbiters])
       maxvalue=data[index]
   
       indexes.append(index)
       maxvals.append(maxvalue)
   
      
       config.writeInput(nb, data,"InputMinIndexMax")
    config.writeReference(nb, maxvals,"MinVals")
    config.writeInputS16(nb, indexes,"MinIndexes")
    return(nb+1)

# Min/Max Abs Tests
def generateMaxAbsTests(config,nb,format,data):
    data = np.abs(data)
    
    indexes=[]
    maxvals=[]

    nbiters = Tools.loopnb(format,Tools.TAILONLY)
    index=np.argmax(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYONLY)
    index=np.argmax(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYANDTAIL)
    index=np.argmax(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    if format == 7:
      # Force max at position 280
  
      nbiters = 280
  
      data = np.zeros(nbiters)
  
      data[nbiters-1] = 0.9 
      data[nbiters-2] = 0.8 
  
      index=np.argmax(data[0:nbiters])
      maxvalue=data[index]
  
      indexes.append(index)
      maxvals.append(maxvalue)

      config.writeInput(nb, data,"InputAbsMaxIndexMax")

    config.writeReference(nb, maxvals,"AbsMaxVals")
    config.writeInputS16(nb, indexes,"AbsMaxIndexes")
    return(nb+1)

def generateMinAbsTests(config,nb,format,data):
    data = np.abs(data)
    
    indexes=[]
    maxvals=[]

    nbiters = Tools.loopnb(format,Tools.TAILONLY)
    index=np.argmin(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYONLY)
    index=np.argmin(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    nbiters = Tools.loopnb(format,Tools.BODYANDTAIL)
    index=np.argmin(data[0:nbiters])
    maxvalue=data[index]

    indexes.append(index)
    maxvals.append(maxvalue)

    if format == 7:
       # Force max at position 280
       nbiters = 280
   
       data = 0.9*np.ones(nbiters)
   
       data[nbiters-1] = 0.0 
       data[nbiters-2] = 0.1 
   
       index=np.argmin(data[0:nbiters])
       maxvalue=data[index]
   
       indexes.append(index)
       maxvals.append(maxvalue)
   
      
       config.writeInput(nb, data,"InputAbsMinIndexMax")
    config.writeReference(nb, maxvals,"AbsMinVals")
    config.writeInputS16(nb, indexes,"AbsMinIndexes")
    return(nb+1)

def averageTest(format,data):
   return(np.average(data))

def powerTest(format,data):
   if format == 31:
       return(np.dot(data,data) / 2**15) # CMSIS is 2.28 format
   elif format == 15:
       return(np.dot(data,data) / 2**33) # CMSIS is 34.30 format
   elif format == 7:
       return(np.dot(data,data) / 2**17) # CMSIS is 18.14 format
   else:
       return(np.dot(data,data))

def rmsTest(format,data):
   return(math.sqrt(np.dot(data,data)/data.size))

def stdTest(format,data):
   return(np.std(data,ddof=1))

def varTest(format,data):
   return(np.var(data,ddof=1))

def generateFuncTests(config,nb,format,data,func,name):

    funcvals=[]

    nbiters = Tools.loopnb(format,Tools.TAILONLY)
    funcvalue=func(format,data[0:nbiters])
    funcvals.append(funcvalue)

    nbiters = Tools.loopnb(format,Tools.BODYONLY)
    funcvalue=func(format,data[0:nbiters])
    funcvals.append(funcvalue)

    nbiters = Tools.loopnb(format,Tools.BODYANDTAIL)
    funcvalue=func(format,data[0:nbiters])
    funcvals.append(funcvalue)

    nbiters = 100
    funcvalue=func(format,data[0:nbiters])
    funcvals.append(funcvalue)

    config.writeReference(nb, funcvals,name)
    return(nb+1)

def generatePowerTests(config,nb,format,data):

    funcvals=[]

    nbiters = Tools.loopnb(format,Tools.TAILONLY)
    funcvalue=powerTest(format,data[0:nbiters])
    funcvals.append(funcvalue)

    nbiters = Tools.loopnb(format,Tools.BODYONLY)
    funcvalue=powerTest(format,data[0:nbiters])
    funcvals.append(funcvalue)

    nbiters = Tools.loopnb(format,Tools.BODYANDTAIL)
    funcvalue=powerTest(format,data[0:nbiters])
    funcvals.append(funcvalue)

    if format==31 or format==15:
      config.writeReferenceQ63(nb, funcvals,"PowerVals")
    elif format==7:
      config.writeReferenceQ31(nb, funcvals,"PowerVals")
    else:
      config.writeReference(nb, funcvals,"PowerVals")
    return(nb+1)

def writeTests(config,nb,format):
    NBSAMPLES = 300
    data1=np.random.randn(NBSAMPLES)
    data2=np.random.randn(NBSAMPLES)
    
    data1 = Tools.normalize(data1)
    data2 = np.abs(data1)

    # Force quantization so that computation of indexes
    # in min/max is coherent between Python and CMSIS.
    # Otherwise there will be normal differences and the test
    # will be displayed as failed.
    if format==31:
       data1=floatRound(data1,31)

    if format==15:
       data1=floatRound(data1,15)

    if format==7:
       data1=floatRound(data1,7)

    config.writeInput(1, data1,"Input")
    config.writeInput(2, data2,"Input")

    nb=generateMaxTests(config,nb,format,data1)
    nb=generateFuncTests(config,nb,format,data2,averageTest,"MeanVals")
    nb=generateMinTests(config,nb,format,data1)
    nb=generatePowerTests(config,nb,format,data1)
    nb=generateFuncTests(config,nb,format,data1,rmsTest,"RmsVals")
    nb=generateFuncTests(config,nb,format,data1,stdTest,"StdVals")
    nb=generateFuncTests(config,nb,format,data1,varTest,"VarVals")
    return(nb)

# We don't want to change ID number of existing tests.
# So new tests have to be added after existing ones
def writeNewsTests(config,nb,format):
    config.setOverwrite(True)
    NBSAMPLES = 300
    data1=np.random.randn(NBSAMPLES)
    
    data1 = Tools.normalize(data1)

    config.writeInput(1, data1,"InputNew")
    nb=generateMaxAbsTests(config,nb,format,data1)
    nb=generateMinAbsTests(config,nb,format,data1)

    config.setOverwrite(False)

def generateBenchmark(config,format):
    NBSAMPLES = 256
    data1=np.random.randn(NBSAMPLES)
    data2=np.random.randn(NBSAMPLES)
    
    data1 = Tools.normalize(data1)
    data2 = np.abs(data1)

    if format==31:
       data1=floatRound(data1,31)

    if format==15:
       data1=floatRound(data1,15)

    if format==7:
       data1=floatRound(data1,7)

    config.writeInput(1, data1,"InputBench")
    config.writeInput(2, data2,"InputBench")


def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Stats","Stats")
    PARAMDIR = os.path.join("Parameters","DSP","Stats","Stats")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configf16=Tools.Config(PATTERNDIR,PARAMDIR,"f16")
    configf64=Tools.Config(PATTERNDIR,PARAMDIR,"f64")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    configq7 =Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    configf32.setOverwrite(False)
    configf16.setOverwrite(False)
    configf64.setOverwrite(False)
    configq31.setOverwrite(False)
    configq15.setOverwrite(False)
    configq7.setOverwrite(False)

    nb=writeTests(configf32,1,0)
    nb=writeF32OnlyTests(configf32,22)
    writeNewsTests(configf32,nb,Tools.F32)

    writeF64OnlyTests(configf64,22)

    nb=writeTests(configq31,1,31)
    writeNewsTests(configq31,nb,Tools.Q31)

    nb=writeTests(configq15,1,15)
    writeNewsTests(configq15,nb,Tools.Q15)

    nb=writeTests(configq7,1,7)
    writeNewsTests(configq7,nb,Tools.Q7)

    nb=writeTests(configf16,1,16)
    nb=writeF16OnlyTests(configf16,22)
    writeNewsTests(configf16,nb,Tools.F16)

    generateBenchmark(configf64, Tools.F64)
    generateBenchmark(configf32, Tools.F32)
    generateBenchmark(configf16, Tools.F16)
    generateBenchmark(configq31, Tools.Q31)
    generateBenchmark(configq15, Tools.Q15)
    generateBenchmark(configq7, Tools.Q7)

if __name__ == '__main__':
  generatePatterns()
