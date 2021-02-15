import os.path
import numpy as np
import itertools
import Tools
from pyquaternion import Quaternion

# mult, multvec, inverse, conjugate, normalize rot2quat, quat2rot , norm
def flattenQuat(l):
    return(np.array([list(x) for x in l]).reshape(4*len(l)))

def flattenRot(l):
    return(np.array([list(x) for x in l]).reshape(9*len(l)))

# q and -q are representing the same rotation.
# So there is an ambiguity for the tests.
# We force the real part of be positive.
def mkQuaternion(mat):
    q=Quaternion(matrix=mat)
    if q.scalar < 0:
        return(-q)
    else:
        return(q)

def writeTests(config,format):
    NBSAMPLES=128

    a=[Quaternion.random() for x in range(NBSAMPLES)]
    b=[Quaternion.random() for x in range(NBSAMPLES)]

    config.writeInput(1, flattenQuat(a))
    config.writeInput(2, flattenQuat(b))

    normTest = [x.norm for x in a]
    config.writeReference(1, normTest)

    inverseTest = [x.inverse for x in a]
    config.writeReference(2, flattenQuat(inverseTest))

    conjugateTest = [x.conjugate for x in a]
    config.writeReference(3, flattenQuat(conjugateTest))

    normalizeTest = [x.normalised for x in a]
    config.writeReference(4, flattenQuat(normalizeTest))

    multTest = [a[i] * b[i] for i in range(NBSAMPLES)]
    config.writeReference(5, flattenQuat(multTest))

    quat2RotTest = [x.rotation_matrix for x in a]
    config.writeReference(6, flattenRot(quat2RotTest))

    config.writeInput(7, flattenRot(quat2RotTest))
    rot2QuatTest = [mkQuaternion(x) for x in quat2RotTest]
    config.writeReference(7, flattenQuat(rot2QuatTest))



    


def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","QuaternionMaths","QuaternionMaths")
    PARAMDIR = os.path.join("Parameters","DSP","QuaternionMaths","QuaternionMaths")
    
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configf16=Tools.Config(PATTERNDIR,PARAMDIR,"f16")
    
    
    writeTests(configf32,0)
    writeTests(configf16,16)


    # Params just as example
    someLists=[[1,3,5],[1,3,5],[1,3,5]]
    
    r=np.array([element for element in itertools.product(*someLists)])
    configf32.writeParam(1, r.reshape(81))

if __name__ == '__main__':
  generatePatterns()
