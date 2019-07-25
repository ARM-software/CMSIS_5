import os.path
import itertools
import Tools
import random
import numpy as np

NBTESTSAMPLES = 10

# Nb vectors for barycenter
NBVECTORS = [4,10,16]

VECDIM = [12,14,20]

def genWsum(config,nb):
    dims=[] 
    inputs=[] 
    weights=[]
    output=[] 
    vecDim = VECDIM[nb % len(VECDIM)]

    dims.append(NBTESTSAMPLES)
    dims.append(vecDim)

    for _ in range(0,NBTESTSAMPLES):
      va = np.random.rand(vecDim)
      vb = np.random.rand(vecDim)
      e = np.sum(va.T * vb) / np.sum(vb) 
      inputs += list(va)
      weights += list(vb)
      output.append(e)
    inputs=np.array(inputs)
    weights=np.array(weights)
    output=np.array(output)
    config.writeInput(nb, inputs,"Inputs")
    config.writeInputS16(nb, dims,"Dims")
    config.writeInput(nb, weights,"Weights")
    config.writeReference(nb, output,"Ref")

def genBarycenter(config,nb):
    dims=[] 
    inputs=[] 
    weights=[]
    output=[] 
    vecDim = VECDIM[nb % len(VECDIM)]
    nbVecs = NBVECTORS[nb % len(NBVECTORS)]

    dims.append(NBTESTSAMPLES)
    dims.append(nbVecs)
    dims.append(vecDim)

    for _ in range(0,NBTESTSAMPLES):
      
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

def writeTests(config):
    genBarycenter(config,1)
    genWsum(config,2)

PATTERNDIR = os.path.join("Patterns","DSP","Support","Support")
PARAMDIR = os.path.join("Parameters","DSP","Support","Support")

configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")

writeTests(configf32)