import os.path
import itertools
import Tools
import random
import numpy as np
import scipy.special as sp

NBTESTSAMPLES = 500

def softmax(v):
  m = sp.softmax(v)
  return(np.argmax(m)+1)

def writeTest(config,nb,vecDim):
    dims=[] 
    inputsA=[]     
    outputs=[] 
    outputsSamples = []


    dims.append(NBTESTSAMPLES)
    dims.append(vecDim)


    for _ in range(0,NBTESTSAMPLES):
      va = np.abs(np.random.randn(vecDim))
      va = va / np.sum(va)

      r = sp.softmax(va)
      outputsSamples += list(r)
      outputs.append(np.argmax(r)+1)
      inputsA += list(va) 


    inputsA=np.array(inputsA)
    outputs=np.array(outputs)
    outputsSamples=np.array(outputsSamples)
    
    config.writeInput(nb, inputsA,"InputA")
    config.writeInputS16(nb, dims,"Dims")

    config.writeReferenceS16(nb, outputs,"Ref")
    config.writeReference(nb, outputsSamples,"Samples")

   


def writeTests(config):
    writeTest(config,1,21)

def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","NN","Softmax",)
    PARAMDIR = os.path.join("Parameters","NN","Softmax")
    
    configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")
    
    writeTests(configq7)


if __name__ == '__main__':
  generatePatterns()