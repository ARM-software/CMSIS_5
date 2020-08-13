import os.path
import itertools
import Tools
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB

def printS(a):
    print("Interpreter[\"Number\"][\"%.9g\"]" % a,end="")

def printV(v):
    start = False
    print("{",end="")
    for r in v:
        if start:
            print(",",end="")
        start = True
        printS(r)
    print("}",end="")

def printM(v):
    start = False
    print("{",end="")
    for r in v:
        if start:
            print(",",end="")
        start = True
        printV(r)
    print("}",end="")

NBTESTSAMPLES = 10

VECDIM = [12,14,20]
BAYESCLASSES= [3,5,4]

NBTRAININGSAMPLES = 30
    
# Distance between the two centers (training vectors are gaussianly 
# distributed around the centers)
CENTER_DISTANCE = 1

TRAININGRATIO = 6.0
PREDICTRATIO = 12.0

# Generate random points distributed around one cluster.
# Cluster are on each axis like (1,0,0,0), (0,1,0,0), (0,0,1,0) etc ...
def newRandomVector(nbClasses,vecDim,ratio):
        v = np.random.randn(vecDim)
        v = v * CENTER_DISTANCE/2.0/ratio
        c = np.random.choice(range(0,nbClasses))
        c0 = np.zeros(vecDim)
        c1 = np.copy(c0)
        c1[c] = c0[0] + CENTER_DISTANCE
        return((v + c1).tolist(),c)

def trainGaussian(nbClasses,vecDim):
        inputs=[] 
        outputs=[]

        # Generate test patterns for this classifier
        for i in range(0,NBTRAININGSAMPLES):
            v,c=newRandomVector(nbClasses,vecDim,TRAININGRATIO)

            inputs.append(v)
            outputs.append(c)

        gnb = GaussianNB()
        gnb.fit(inputs, outputs)
        return(gnb)

def generateNewTest(config,nb):
    dims=[] 
    inputs=[] 
    referenceproba=[] 
    referencepredict=[]
    params=[]
    dims.append(NBTESTSAMPLES)
    classNb = BAYESCLASSES[nb % len(BAYESCLASSES)]
    vecDim = VECDIM[nb % len(VECDIM)]
    dims.append(classNb)
    dims.append(vecDim)
    # Train a classifier for a given vector dimension and
    # given number of classes
    gb = trainGaussian(classNb,vecDim)
    params += list(np.reshape(gb.theta_,np.size(gb.theta_)))
    params += list(np.reshape(gb.sigma_,np.size(gb.sigma_)))
    params += list(np.reshape(gb.class_prior_,np.size(gb.class_prior_)))
    params.append(gb.epsilon_)

    #print("theta=",end="")
    #printM(gb.theta_)
    #print(";",end="")
    #
    #print("sigma=",end="")
    #printM(gb.sigma_)
    #print(";",end="")
    #
    #print("prior=",end="")
    #printV(gb.class_prior_)
    #print(";",end="")
    #
    #print("epsilon=",end="")
    #printS(gb.epsilon_)
    #print(";",end="")

    #print(classNb,vecDim)
    for _ in range(0,NBTESTSAMPLES):
        # Generate a test pattern for this classifier
        v,c=newRandomVector(classNb,vecDim,PREDICTRATIO)
        inputs += v
        #print("inputs=",end="")
        #printV(v)
        #print(";",end="")
       
        y_pred = gb.predict([v])
        referencepredict.append(y_pred[0])
       
        probas = gb._joint_log_likelihood([v])
        probas = probas[0]
        referenceproba += list(probas)
        

    inputs = np.array(inputs)
    params = np.array(params)
    referenceproba = np.array(referenceproba)
    referencepredict = np.array(referencepredict)
    dims = np.array(dims)

    config.writeInput(nb, inputs,"Inputs")
    config.writeInputS16(nb, dims,"Dims")
    config.writeReference(nb, referenceproba,"Probas")
    config.writeReferenceS16(nb, referencepredict,"Predicts")
    config.writeReference(nb, params,"Params")
    
    #print(inputs)
    #print(dims)
    #print(referencepredict)
    #print(referenceproba)
    #print(params)


def writeTests(config):
    generateNewTest(config,1)

def writeBenchmark(config):
    someLists=[VECDIM,BAYESCLASSES]
    
    r=np.array([element for element in itertools.product(*someLists)])
    nbtests=len(VECDIM)*len(BAYESCLASSES)*2
    config.writeParam(2, r.reshape(nbtests))

    params=[]
    inputs=[] 
    referencepredict=[]
    dims=[] 
    nbin=0
    nbparam=0;

    for vecDim, classNb in r:
        gb = trainGaussian(classNb,vecDim)
        p = []
        p += list(np.reshape(gb.theta_,np.size(gb.theta_)))
        p += list(np.reshape(gb.sigma_,np.size(gb.sigma_)))
        p += list(np.reshape(gb.class_prior_,np.size(gb.class_prior_)))
        p.append(gb.epsilon_)

        params += p
        dims += [nbin,nbparam]
        nbparam = nbparam + len(p)

        v,c=newRandomVector(classNb,vecDim,PREDICTRATIO)
        inputs += v

        nbin = nbin + len(v)

        y_pred = gb.predict([v])
        referencepredict.append(y_pred[0])
       

    inputs = np.array(inputs)
    params = np.array(params)
    referencepredict = np.array(referencepredict)
    dims = np.array(dims)

    config.writeInput(2, inputs,"Inputs")
    config.writeReferenceS16(2, referencepredict,"Predicts")
    config.writeReference(2, params,"Params")
    config.writeInputS16(2, dims,"DimsBench")


def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Bayes","Bayes")
    PARAMDIR = os.path.join("Parameters","DSP","Bayes","Bayes")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configf16=Tools.Config(PATTERNDIR,PARAMDIR,"f16")
    
    writeTests(configf32)
    writeTests(configf16)

    writeBenchmark(configf32)
    writeBenchmark(configf16)

if __name__ == '__main__':
  generatePatterns()