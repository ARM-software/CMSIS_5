import os.path
import numpy as np
import itertools
import Tools
import statsmodels.tsa.stattools

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

def cartesian(*somelists):
   r=[]
   for element in itertools.product(*somelists):
       r.append(element)
   return(r)

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def writeTests(config,format):

    config.setOverwrite(False)

    NBSAMPLES=128

    inputsA=np.random.randn(NBSAMPLES)
    inputsB=np.random.randn(NBSAMPLES)

    inputsA = Tools.normalize(inputsA)
    inputsB = Tools.normalize(inputsB)

    if format==31:
      # To avoid overflow. There is no saturation in CMSIS code for Q31 conv/corr
      inputsA = inputsA / 16
      inputsB = inputsB / 16
    

    config.writeInput(1, inputsA,"InputsA")
    config.writeInput(1, inputsB,"InputsB")

    a = [1,2,3,Tools.loopnb(format,Tools.TAILONLY),
    Tools.loopnb(format,Tools.BODYONLY),
    Tools.loopnb(format,Tools.BODYANDTAIL)
    ]

    a = list(np.unique(np.array(a)))

    if format == 15:
       nbs = [(14, 15), (14, 16), (14, 17), (14, 18), (14, 33), (15, 15), 
              (15, 16), (15, 17), (15, 18), (15, 33), (16, 15), (16, 16), 
              (16, 17), (16, 18), (16, 33), (17, 15), (17, 16), (17, 17), 
              (17, 18), (17, 33), (32, 15), (32, 16), (32, 17), (32, 18), (32, 33)]
    elif format == 7 :
       nbs = [(30, 31), (30, 32), (30, 33), (30, 34), (30, 49), (31, 31), 
              (31,32), (31, 33), (31, 34), (31, 49), (32, 31), (32, 32), 
              (32, 33), (32,34), (32, 49), (33, 31), (33, 32), (33, 33), (33, 34), 
              (33, 49), (48,31), (48, 32), (48, 33), (48, 34), (48, 49)]
    else:
       nbs = [(4, 1), (4, 2), (4, 3), (4, 8), (4, 11), (5, 1), (5, 2), (5, 3), (5, 8), (5, 11), (6, 1), (6, 2), (6, 3), (6, 8), (6, 11), (9, 1), (9, 2), 
              (9, 3), (9, 8), (9, 11), (10, 1), (10, 2), (10, 3), (10, 8), (10, 11), (11, 1), (11, 2), (11, 3), (11, 8), (11, 11), (12, 1), (12, 2), 
              (12, 3), (12, 8), (12, 11), (13, 1), (13, 2), (13, 3), (13, 8), (13, 11)]

    nbTest = 1

    for (na,nb) in nbs:
        #print(na,nb)

        ref = np.correlate(inputsA[0:na],inputsB[0:nb],"full")
        if na > nb:
           padding = na - nb
           z = np.zeros(padding)
           ref = np.concatenate((z,ref))
        else:
           padding = nb - na
           z = np.zeros(padding)
           ref = np.concatenate((ref,z))
        config.writeReference(nbTest, ref)
        nbTest = nbTest + 1

    for (na,nb) in nbs:
        #print(na,nb)

        ref = np.convolve(inputsA[0:na],inputsB[0:nb],"full")
        config.writeReference(nbTest, ref)
        nbTest = nbTest + 1

    # Levinson durbin tests

    a = [Tools.loopnb(format,Tools.TAILONLY),
    Tools.loopnb(format,Tools.BODYONLY),
    Tools.loopnb(format,Tools.BODYANDTAIL),
    ]

    a = list(np.unique(np.array(a)))

    #a = [3]

    # Errors of each levinson durbin test
    err=[]

    errTestID = nbTest

    for na in a:
      
      s = np.random.randn(na+1)
      s = Tools.normalize(s)
      phi = autocorr(s)

      phi = Tools.normalize(phi)

      config.writeInput(nbTest, phi,"InputPhi")

      sigmav,arcoef,pacf,sigma,phi=statsmodels.tsa.stattools.levinson_durbin(phi,nlags=na,isacov=True)
      
      err.append(sigmav)
      
      config.writeReference(nbTest, arcoef)
      nbTest = nbTest + 1

      config.writeReference(errTestID, err,"LDErrors")

    # Partial convolutions
    config.setOverwrite(True)

    inputsA=np.random.randn(NBSAMPLES)
    inputsB=np.random.randn(NBSAMPLES)

    inputsA = Tools.normalize(inputsA)
    inputsB = Tools.normalize(inputsB)

    config.writeInput(2, inputsA,"InputsA")
    config.writeInput(2, inputsB,"InputsB")

    (na,nb) = (6, 8) 
    # First = 3
    numPoints=4
    ref = np.convolve(inputsA[0:na],inputsB[0:nb],"full")
    
    first=3
    config.writeReference(nbTest, ref[first:first+numPoints])
    nbTest = nbTest + 1

    first=9
    config.writeReference(nbTest, ref[first:first+numPoints])
    nbTest = nbTest + 1

    first=7
    config.writeReference(nbTest, ref[first:first+numPoints])
    nbTest = nbTest + 1



    

    
def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Filtering","MISC","MISC")
    PARAMDIR = os.path.join("Parameters","DSP","Filtering","MISC","MISC")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configf16=Tools.Config(PATTERNDIR,PARAMDIR,"f16")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")
    configq7=Tools.Config(PATTERNDIR,PARAMDIR,"q7")

    
    writeTests(configf32,0)
    writeTests(configf16,16)
    writeTests(configq31,31)
    writeTests(configq15,15)
    writeTests(configq7,7)


if __name__ == '__main__':
  generatePatterns()
