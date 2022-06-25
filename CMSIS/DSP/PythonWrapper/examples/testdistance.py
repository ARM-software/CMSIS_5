import cmsisdsp as dsp 
import numpy as np
import scipy.spatial.distance as d 
from numpy.testing import assert_allclose

a=[1,2,3]
b=[1,5,2]

print("\nBray-Curtis")
ref=d.braycurtis(a,b)
res=dsp.arm_braycurtis_distance_f32(a,b)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)


print("\nCanberra")
ref=d.canberra(a,b)
res=dsp.arm_canberra_distance_f32(a,b)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nChebyshev")
ref=d.chebyshev(a,b)
res=dsp.arm_chebyshev_distance_f32(a,b)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

res=dsp.arm_chebyshev_distance_f64(a,b)
print(res)
assert_allclose(ref,res,1e-10)

print("\nCity Block")
ref=d.cityblock(a,b)
res=dsp.arm_cityblock_distance_f32(a,b)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

res=dsp.arm_cityblock_distance_f64(a,b)
print(res)
assert_allclose(ref,res,1e-10)

print("\nCorrelation")
ref=d.correlation(a,b)
res=dsp.arm_correlation_distance_f32(a,b)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nCosine")
ref=d.cosine(a,b)
res=dsp.arm_cosine_distance_f32(a,b)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

res=dsp.arm_cosine_distance_f64(a,b)
print(res)
assert_allclose(ref,res,1e-10)

print("\nEuclidean")
ref=d.euclidean(a,b)
res=dsp.arm_euclidean_distance_f32(a,b)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

res=dsp.arm_euclidean_distance_f64(a,b)
print(res)
assert_allclose(ref,res,1e-10)

print("\nJensen-Shannon")
pa=a/np.sum(a)
pb=b/np.sum(b)
ref=d.jensenshannon(pa,pb)
res=dsp.arm_jensenshannon_distance_f32(pa,pb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nMinkowski")
w=3
ref=d.minkowski(a,b,w)
res=dsp.arm_minkowski_distance_f32(a,b,w)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

# Int distance
# For CMSIS-DSP the bool must be packed as bit arrays

# Pack an array of boolean into uint32
def packset(a):
    b = np.packbits(a)
    newSize = int(np.ceil(b.shape[0] / 4.0)) * 4
    c = np.copy(b)
    c.resize(newSize)
    #print(c)
    vecSize = round(newSize/4)
    c=c.reshape(vecSize,4)
    #print(c)
    r = np.zeros(vecSize)
    result = []
    for i in range(0,vecSize):
        #print(c[i,:])
        #print("%X %X %X %X" % (c[i,0],c[i,1],c[i,2],c[i,3]))
        d = (c[i,0] << 24) | (c[i,1] << 16) | (c[i,2] << 8) | c[i,3] 
        result.append(np.uint32(d))
    return(result) 

nb = 34
va = np.random.choice([0,1],nb)
# Array of word32 containing all of our bits
pva = packset(va)


vb = np.random.choice([0,1],nb)
# Array of word32 containing all of our bits
pvb = packset(vb)

print("\nDice")
ref=d.dice(va,vb)
res=dsp.arm_dice_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nHamming")
ref=d.hamming(va,vb)
res=dsp.arm_hamming_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nJaccard-Needham")
ref=d.jaccard(va,vb)
res=dsp.arm_jaccard_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nKulsinski")
ref=d.kulsinski(va,vb)
res=dsp.arm_kulsinski_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nRogers-Tanimoto")
ref=d.rogerstanimoto(va,vb)
res=dsp.arm_rogerstanimoto_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nRussell-Rao")
ref=d.russellrao(va,vb)
res=dsp.arm_russellrao_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nSokal-Michener")
ref=d.sokalmichener(va,vb)
res=dsp.arm_sokalmichener_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nSokal-Sneath")
ref=d.sokalsneath(va,vb)
res=dsp.arm_sokalsneath_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)

print("\nYule")
ref=d.yule(va,vb)
res=dsp.arm_yule_distance(pva,pvb,nb)
print(ref)
print(res)
assert_allclose(ref,res,1e-6)
