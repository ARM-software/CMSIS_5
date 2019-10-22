import os.path
import itertools
import Tools
import random
import numpy as np
import scipy.spatial

NBTESTSAMPLES = 10

VECDIM = [35,14,20]

def euclidean(xa,xb):
        r = scipy.spatial.distance.euclidean(xa,xb)
        return(r)

def braycurtis(xa,xb):
        r = scipy.spatial.distance.braycurtis(xa,xb)
        return(r)

def canberra(xa,xb):
        r = scipy.spatial.distance.canberra(xa,xb)
        return(r)

def chebyshev(xa,xb):
        r = scipy.spatial.distance.chebyshev(xa,xb)
        return(r)

def cityblock(xa,xb):
        r = scipy.spatial.distance.cityblock(xa,xb)
        return(r)

def correlation(xa,xb):
        r = scipy.spatial.distance.correlation (xa,xb)
        return(r)

def cosine(xa,xb):
        r = scipy.spatial.distance.cosine (xa,xb)
        return(r)

def jensenshannon(xa,xb):
        r = scipy.spatial.distance.jensenshannon (xa,xb)
        return(r)

def minkowski (xa,xb,dim):
        r = scipy.spatial.distance.minkowski(xa,xb,p=dim)
        return(r)

def dice(xa,xb):
        r = scipy.spatial.distance.dice (xa,xb)
        return(r)

def hamming(xa,xb):
        r = scipy.spatial.distance.hamming (xa,xb)
        return(r)

def jaccard(xa,xb):
        r = scipy.spatial.distance.jaccard (xa,xb)
        return(r)

def kulsinski(xa,xb):
        r = scipy.spatial.distance.kulsinski (xa,xb)
        return(r)

def rogerstanimoto(xa,xb):
        r = scipy.spatial.distance.rogerstanimoto (xa,xb)
        return(r)

def russellrao(xa,xb):
        r = scipy.spatial.distance.russellrao (xa,xb)
        return(r)

def sokalmichener(xa,xb):
        r = scipy.spatial.distance.sokalmichener (xa,xb)
        return(r)

def sokalsneath(xa,xb):
        r = scipy.spatial.distance.sokalsneath (xa,xb)
        return(r)

def yule(xa,xb):
        r = scipy.spatial.distance.yule (xa,xb)
        return(r)

def writeFTest(config,funcList):
    dims=[] 
    dimsM=[]
    inputsA=[] 
    inputsB=[]
    inputsAJ=[] 
    inputsBJ=[]
    outputs=[] 
    outputMin=[] 
    outputJen=[] 
    for i in range(0,len(funcList)):
        outputs.append([])

    vecDim = VECDIM[0]

    dims.append(NBTESTSAMPLES)
    dims.append(vecDim)

    dimsM.append(NBTESTSAMPLES)
    dimsM.append(vecDim)

    for _ in range(0,NBTESTSAMPLES):
      normDim = np.random.choice([2,3,4])
      dimsM.append(normDim)
      va = np.random.randn(vecDim)
      # Normalization for distance assuming probability distribution in entry
      vb = np.random.randn(vecDim)
      for i in range(0,len(funcList)):
        func = funcList[i]
        outputs[i].append(func(va,vb))
      outputMin.append(minkowski(va,vb,normDim))

      inputsA += list(va) 
      inputsB += list(vb)

      va = np.abs(va)
      va = va / np.sum(va)

      vb = np.abs(vb)
      vb = vb / np.sum(vb)

      inputsAJ += list(va) 
      inputsBJ += list(vb)
      outputJen.append(jensenshannon(va,vb)) 


    inputsA=np.array(inputsA)
    inputsB=np.array(inputsB)
    for i in range(0,len(funcList)):
      outputs[i]=np.array(outputs[i])
    
    config.writeInput(1, inputsA,"InputA")
    config.writeInput(1, inputsB,"InputB")
    config.writeInput(8, inputsAJ,"InputA")
    config.writeInput(8, inputsBJ,"InputB")
    config.writeInputS16(1, dims,"Dims")
    config.writeInputS16(9, dimsM,"Dims")

    for i in range(0,len(funcList)):
       config.writeReference(i+1, outputs[i],"Ref")

    config.writeReference(8, outputJen,"Ref")
    config.writeReference(9, outputMin,"Ref")

def writeBTest(config,funcList):
    dims=[] 
    inputsA=[] 
    inputsB=[]
    outputs=[] 
    for i in range(0,len(funcList)):
        outputs.append([])

    vecDim = VECDIM[0]

    dims.append(NBTESTSAMPLES)
    dims.append(vecDim)
    va = np.random.choice([0,1],vecDim)
    # Number of word32 containing all of our bits
    pva = Tools.packset(va)
    dims.append(len(pva))

    for _ in range(0,NBTESTSAMPLES):
      va = np.random.choice([0,1],vecDim)
      vb = np.random.choice([0,1],vecDim)
      # Boolean arrays are packed for the C code
      pva = Tools.packset(va)
      pvb = Tools.packset(vb)
      for i in range(0,len(funcList)):
        func = funcList[i]
        outputs[i].append(func(va,vb))

      inputsA += pva 
      inputsB += pvb

    inputsA=np.array(inputsA)
    inputsB=np.array(inputsB)
    for i in range(0,len(funcList)):
      outputs[i]=np.array(outputs[i])
    
    config.writeInput(1, inputsA,"InputA")
    config.writeInput(1, inputsB,"InputB")
    config.writeInputS16(1, dims,"Dims")

    for i in range(0,len(funcList)):
       config.writeReferenceF32(i+1, outputs[i],"Ref")

def writeFTests(config):
    writeFTest(config,[braycurtis,canberra,chebyshev,cityblock,correlation,cosine,euclidean])

def writeBTests(config):
    writeBTest(config,[dice,hamming,jaccard,kulsinski,rogerstanimoto,russellrao,sokalmichener,sokalsneath,yule])

def  generatePatterns():
     PATTERNDIR = os.path.join("Patterns","DSP","Distance","Distance")
     PARAMDIR = os.path.join("Parameters","DSP","Distance","Distance")
     
     configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
     configu32=Tools.Config(PATTERNDIR,PARAMDIR,"u32")
     
     writeFTests(configf32)
     writeBTests(configu32)

if __name__ == '__main__':
  generatePatterns()