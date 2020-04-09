import os.path
import itertools
import Tools
import random
import numpy as np
from scipy import interpolate

NBTESTSAMPLES = 10

# Nb vectors for barycenter
NBVECTORS = [4,10,16]

VECDIM = [12,14,20]

def genWsum(config,nb):
    DIM=50
    inputs=[] 
    weights=[]
    output=[] 


    va = np.random.rand(DIM)
    vb = np.random.rand(DIM)
    inputs += list(va)
    weights += list(vb)

    nbiters = Tools.loopnb(0,Tools.TAILONLY)
    e = np.sum(va[0:nbiters].T * vb[0:nbiters]) / np.sum(vb[0:nbiters]) 
    output.append(e)

    nbiters = Tools.loopnb(0,Tools.BODYONLY)
    e = np.sum(va[0:nbiters].T * vb[0:nbiters]) / np.sum(vb[0:nbiters]) 
    output.append(e)

    nbiters = Tools.loopnb(0,Tools.BODYANDTAIL)
    e = np.sum(va[0:nbiters].T * vb[0:nbiters]) / np.sum(vb[0:nbiters]) 
    output.append(e)

    inputs=np.array(inputs)
    weights=np.array(weights)
    output=np.array(output)
    config.writeInput(nb, inputs,"Inputs")
    config.writeInput(nb, weights,"Weights")
    config.writeReference(nb, output,"Ref")

def genBarycenter(config,nb,nbTests,nbVecsArray,vecDimArray):
    dims=[] 
    inputs=[] 
    weights=[]
    output=[] 

    dims.append(nbTests)
    

    for i in range(0,nbTests):
      nbVecs = nbVecsArray[i % len(nbVecsArray)]
      vecDim = vecDimArray[i % len(vecDimArray)]
      dims.append(nbVecs )
      dims.append(vecDim)

      vecs = []
      b = np.zeros(vecDim)
      coefs = np.random.rand(nbVecs)
  
      for i in range(nbVecs):
            va = np.random.rand(vecDim)
            b += va * coefs[i]
            vecs += list(va)
                     
      b = b / np.sum(coefs)
     
      inputs += list(vecs)
      weights += list(coefs)
      output += list(b)
    inputs=np.array(inputs)
    weights=np.array(weights)
    output=np.array(output)
    config.writeInput(nb, inputs,"Inputs")
    config.writeInputS16(nb, dims,"Dims")
    config.writeInput(nb, weights,"Weights")
    config.writeReference(nb, output,"Ref")

def writeTestsF32(config):
    NBSAMPLES=256

    va = np.random.rand(NBSAMPLES)
    va = Tools.normalize(va)
    config.writeInput(1,va,"Samples")

    config.writeInputQ15(3,va,"Samples")
    config.writeInputQ31(4,va,"Samples")
    config.writeInputQ7(5,va,"Samples")

    # This is for benchmarking the weighted sum and we use only one test pattern
    genWsum(config,6)
    


def writeTestsQ31(config):
    NBSAMPLES=256

    va = np.random.rand(NBSAMPLES)
    va = Tools.normalize(va)
    config.writeInputF32(1,va,"Samples")
    config.writeInputQ15(3,va,"Samples")
    config.writeInput(4,va,"Samples")
    config.writeInputQ7(5,va,"Samples")


def writeTestsQ15(config):
    NBSAMPLES=256

    va = np.random.rand(NBSAMPLES)
    va = Tools.normalize(va)
    config.writeInputF32(1,va,"Samples")
    config.writeInput(3,va,"Samples")
    config.writeInputQ31(4,va,"Samples")
    config.writeInputQ7(5,va,"Samples")

def writeTestsQ7(config):
    NBSAMPLES=256

    va = np.random.rand(NBSAMPLES)
    va = Tools.normalize(va)
    config.writeInputF32(1,va,"Samples")
    config.writeInputQ15(3,va,"Samples")
    config.writeInputQ31(4,va,"Samples")
    config.writeInput(5,va,"Samples")

def writeBarTests(config):
    # For testing
    NBSAMPLES = 10
    nbVecsArray = [4,8,9] 
    vecDimArray = [4,4,4,8,8,8,9,9,9]
    genBarycenter(config,1,NBTESTSAMPLES,nbVecsArray,vecDimArray)

    # For benchmarks
    va = np.random.rand(128*15)
    vb = np.random.rand(128)
    config.writeInput(1,va,"Samples")
    config.writeInput(1,vb,"Coefs")

def writeTests2(config, format):

    data = np.random.randn(11)    
    data = Tools.normalize(data)
    config.writeInput(7, data)
    ref = np.sort(data)
    config.writeReference(7, ref)

    data = np.random.randn(16)
    data = Tools.normalize(data)
    config.writeInput(8, data)
    ref = np.sort(data)
    config.writeReference(8, ref)

    data = np.random.randn(32)
    data = Tools.normalize(data)
    config.writeInput(9, data)
    ref = np.sort(data)
    config.writeReference(9, ref)

    data = np.full((16), np.random.randn(1))
    data = Tools.normalize(data)
    config.writeInput(10, data)
    ref = np.sort(data)
    config.writeReference(10, ref)

    x = [0,3,10,20]
    config.writeInput(11,x,"InputX")
    y = [0,9,100,400]
    config.writeInput(11,y,"InputY")
    xnew = np.arange(0,20,1)
    config.writeInput(11,xnew,"OutputX")
    ynew = interpolate.CubicSpline(x,y)
    config.writeReference(11, ynew(xnew))

    x = np.arange(0, 2*np.pi+np.pi/4, np.pi/4)
    config.writeInput(12,x,"InputX")
    y = np.sin(x)
    config.writeInput(12,y,"InputY")
    xnew = np.arange(0, 2*np.pi+np.pi/16, np.pi/16)
    config.writeInput(12,xnew,"OutputX")
    ynew = interpolate.CubicSpline(x,y,bc_type="natural")
    config.writeReference(12, ynew(xnew))

    x = [0,3,10]
    config.writeInput(13,x,"InputX")
    y = x
    config.writeInput(13,y,"InputY")
    xnew = np.arange(-10,20,1)
    config.writeInput(13,xnew,"OutputX")
    ynew = interpolate.CubicSpline(x,y)
    config.writeReference(13, ynew(xnew))

def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Support","Support")
    PARAMDIR = os.path.join("Parameters","DSP","Support","Support")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    writeTestsF32(configf32)
    writeTestsQ31(configq31)
    writeTestsQ15(configq15)
    writeTestsQ7(configq7)

    writeTests2(configf32,0)

    
    # For benchmarking we need to vary number of vectors and vector dimension separately
    PATTERNBARDIR = os.path.join("Patterns","DSP","SupportBar")
    PARAMBARDIR = os.path.join("Parameters","DSP","SupportBar")
    
    configBarf32=Tools.Config(PATTERNBARDIR,PARAMBARDIR,"f32")
    
    writeBarTests(configBarf32)

if __name__ == '__main__':
  generatePatterns()
