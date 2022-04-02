import cmsisdsp as dsp 
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import math

a=[1,4,2,6,7,0,-3,5]
ref=sorted(a)
print(ref)

SORT_BITONIC=0
SORT_BUBBLE=1
SORT_HEAP=2
SORT_INSERTION=3
SORT_QUICK=4
SORT_SELECTION=5


SORT_DESCENDING = 0
SORT_ASCENDING = 1

sortinst=dsp.arm_sort_instance_f32()

for mode in range(6):
    dsp.arm_sort_init_f32(sortinst,mode,SORT_ASCENDING)
    res=dsp.arm_sort_f32(sortinst,a)
    print(res)
    assert (res==ref).all()

print("")
ref.reverse()
print(ref)

for mode in range(6):
    # Problem with bitonic probably in the C code
    if mode > 0:
       dsp.arm_sort_init_f32(sortinst,mode,SORT_DESCENDING)
       res=dsp.arm_sort_f32(sortinst,a)
       print(res)
       assert (res==ref).all()

print("Spline")

x = np.arange(0, 2*np.pi+np.pi/4, np.pi/4)
y = np.sin(x)
xnew = np.arange(0, 2*np.pi+np.pi/16, np.pi/16)
ynew = CubicSpline(x,y,bc_type="natural")
yref=ynew(xnew)
print(yref)

splineInst = dsp.arm_spline_instance_f32()
dsp.arm_spline_init_f32(splineInst,0,x,y)
yres=dsp.arm_spline_f32(splineInst,xnew)
print(yres)

assert_allclose(yref,yres,1e-6,1e-6)

print("Bayes")
# Reusing example from https://developer.arm.com/documentation/102052/0000/Train-your-Bayesian-estimator-with-scikit-learn

NBVECS = 100
VECDIM = 2

# 3 cluster of points are generated (3 classes)
ballRadius = 1.0
x1 = [1.5, 1] +  ballRadius * np.random.randn(NBVECS,VECDIM)
x2 = [-1.5, 1] + ballRadius * np.random.randn(NBVECS,VECDIM)
x3 = [0, -3] + ballRadius * np.random.randn(NBVECS,VECDIM)

# All points are concatenated
X_train=np.concatenate((x1,x2,x3))

# The classes are 0,1 and 2.
Y_train=np.concatenate((np.zeros(NBVECS),np.ones(NBVECS),2*np.ones(NBVECS)))

gnb = GaussianNB()
gnb.fit(X_train, Y_train)

src1=[1.5,1.0]
src2=[-1.5,1]
src3=[0,-3]
ref1 = gnb.predict([src1])
print(ref1)

ref2 = gnb.predict([src2])
print(ref2)

ref3 = gnb.predict([src3])
print(ref3)

#print(gnb.predict_log_proba([src]))

theta=list(np.reshape(gnb.theta_,np.size(gnb.theta_)))

# Gaussian variances
sigma=list(np.reshape(gnb.var_,np.size(gnb.var_)))

# Class priors
prior=list(np.reshape(gnb.class_prior_,np.size(gnb.class_prior_)))

epsilon=gnb.epsilon_

bayesInst = dsp.arm_gaussian_naive_bayes_instance_f32(
    vectorDimension=VECDIM,numberOfClasses=3,
    theta=theta,sigma=sigma,classPriors=prior,epsilon=epsilon)

_,res1=dsp.arm_gaussian_naive_bayes_predict_f32(bayesInst,src1)
print(res1)

_,res2=dsp.arm_gaussian_naive_bayes_predict_f32(bayesInst,src2)
print(res2)

_,res3=dsp.arm_gaussian_naive_bayes_predict_f32(bayesInst,src3)
print(res3)

assert res1 == ref1
assert res2 == ref2
assert res3 == ref3

print("SVM")

NBVECS = 100
VECDIM = 2

ballRadius = 0.5
x = ballRadius * np.random.randn(NBVECS, 2)

angle = 2.0 * math.pi * np.random.randn(1, NBVECS)
radius = 3.0 + 0.1 * np.random.randn(1, NBVECS)

xa = np.zeros((NBVECS,2))
xa[:, 0] = radius * np.cos(angle)
xa[:, 1] = radius * np.sin(angle)

X_train = np.concatenate((x, xa))
Y_train = np.concatenate((np.zeros(NBVECS), np.ones(NBVECS)))

clf = svm.SVC(kernel='poly', gamma='auto', coef0=1.1)
clf.fit(X_train, Y_train)

test1 = np.array([0.4,0.1])
test1 = test1.reshape(1,-1)

refpredicted1 = clf.predict(test1)
print(refpredicted1)

test2 = np.array([3.1,0.1])
test2 = test2.reshape(1,-1)

refpredicted2 = clf.predict(test2)
print(refpredicted2)

supportShape = clf.support_vectors_.shape

nbSupportVectors = supportShape[0]
vectorDimensions = supportShape[1]

degree=clf.degree
coef0=clf.coef0
gamma=clf._gamma

intercept=clf.intercept_

dualCoefs = clf.dual_coef_
dualCoefs = dualCoefs.reshape(nbSupportVectors)
supportVectors = clf.support_vectors_
supportVectors = supportVectors.reshape(nbSupportVectors * VECDIM)

svmInst=dsp.arm_svm_polynomial_instance_f32() 
dsp.arm_svm_polynomial_init_f32(svmInst,nbSupportVectors,vectorDimensions,
    intercept,dualCoefs,supportVectors,
    [0,1],degree,coef0,gamma)

test1 = np.array([0.4,0.1])
predicted1 = dsp.arm_svm_polynomial_predict_f32(svmInst,test1)
print(predicted1)

test2 = np.array([3.1,0.1])
predicted2 = dsp.arm_svm_polynomial_predict_f32(svmInst,test2)
print(predicted2)

assert predicted1==refpredicted1 
assert predicted2==refpredicted2 