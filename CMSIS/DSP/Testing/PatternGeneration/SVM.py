import os.path
import itertools
import Tools
from sklearn import svm
import random
import numpy as np

# Number of vectors to test for each test
NBTESTSAMPLE = 100
# Dimension of the vectors
VECDIM = 10

# Number of vectors for training
NBVECTORS=10
# Distance between the two centers (training vectors are gaussianly 
# distributed around the centers)
CENTER_DISTANCE = 1

# SVM KIND
LINEAR=1
POLY=2
RBF=3
SIGMOID=4

C0 = np.zeros((1,VECDIM))
C1 = np.copy(C0)
C1[0,0] = C1[0,0] + CENTER_DISTANCE

# Data for training

X = []
Xone = []
y = []

class1 = 0 
class2 = 1

for i in range(NBVECTORS):
    v = np.random.randn(1,VECDIM)
    v = v * CENTER_DISTANCE/2.0/10 
    # 2 classes are needed
    if i == 0:
        c = 0 
    elif i == 1:
        c = 1
    else:
        c = np.random.choice([0,1])
    if (c == 0):
        v = v + C0
        y.append(class1)
    else:
        v = v + C1 
        y.append(class2)
    if c == 0:
        Xone.append(v[0].tolist())
    X.append(v[0].tolist())

def newSVMTest(config,kind,theclass,clf,nb):
    inputs = [] 
    references = []
    for i in range(NBTESTSAMPLE):
            v = np.random.randn(1,VECDIM)
            v = v * CENTER_DISTANCE/2.0/6.0 
            c = np.random.choice([0,1])
            if (c == 0):
               v = v + C0 
            else:
               v = v + C1 
            inputs.append(v[0].tolist())
            toPredict=[v[0].tolist()]
            references.append(clf.predict(toPredict))
    inputs=np.array(inputs)
    inputs=inputs.reshape(NBTESTSAMPLE*VECDIM)

    config.writeInput(nb, inputs,"Samples")

    references=np.array(references)
    references=references.reshape(NBTESTSAMPLE)

    # Classifier description
    supportShape = clf.support_vectors_.shape

    nbSupportVectors=supportShape[0]
    vectorDimensions=supportShape[1]
    intercept = np.array(clf.intercept_)
    dualCoefs=clf.dual_coef_ 
    dualCoefs=dualCoefs.reshape(nbSupportVectors)
    supportVectors=clf.support_vectors_
    supportVectors = supportVectors.reshape(nbSupportVectors*VECDIM)
    
    if kind == LINEAR:
       dims=np.array([kind,theclass[0],theclass[1],NBTESTSAMPLE,VECDIM,nbSupportVectors])
    elif kind==POLY:
       dims=np.array([kind,theclass[0],theclass[1],NBTESTSAMPLE,VECDIM,nbSupportVectors,clf.degree])
    elif kind==RBF:
       dims=np.array([kind,theclass[0],theclass[1],NBTESTSAMPLE,VECDIM,nbSupportVectors])
    elif kind==SIGMOID:
       dims=np.array([kind,theclass[0],theclass[1],NBTESTSAMPLE,VECDIM,nbSupportVectors])
    
    config.writeInputS16(nb, dims,"Dims")

    if kind == LINEAR:
       params=np.concatenate((supportVectors,dualCoefs,intercept))
    elif kind == POLY:
       coef0 = np.array([clf.coef0])
       gamma = np.array([clf._gamma])
       params=np.concatenate((supportVectors,dualCoefs,intercept,coef0,gamma))
    elif kind == RBF:
       gamma = np.array([clf._gamma])
       params=np.concatenate((supportVectors,dualCoefs,intercept,gamma))
    elif kind == SIGMOID:
       coef0 = np.array([clf.coef0])
       gamma = np.array([clf._gamma])
       params=np.concatenate((supportVectors,dualCoefs,intercept,coef0,gamma))

    config.writeInput(nb, params,"Params")

    config.writeReferenceS32(nb, references,"Reference")


def writeTests(config):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    newSVMTest(config,LINEAR,[class1,class2],clf,1)

    clf = svm.SVC(kernel='poly',gamma='auto', coef0=1.1)
    clf.fit(X, y)
    newSVMTest(config,POLY,[class1,class2],clf,2)

    clf = svm.SVC(kernel='rbf',gamma='auto')
    clf.fit(X, y)
    newSVMTest(config,RBF,[class1,class2],clf,3)

    clf = svm.SVC(kernel='sigmoid',gamma='auto')
    clf.fit(X, y)
    newSVMTest(config,SIGMOID,[class1,class2],clf,4)

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X)
    newSVMTest(config,RBF,[-1,1],clf,5)

    

def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","SVM","SVM")
    PARAMDIR = os.path.join("Parameters","DSP","SVM","SVM")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    
    writeTests(configf32)

if __name__ == '__main__':
  generatePatterns()