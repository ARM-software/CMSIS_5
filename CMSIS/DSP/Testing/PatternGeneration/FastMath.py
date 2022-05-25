import os.path
import numpy as np
import itertools
import Tools
import math

import numpy as np

def q31accuracy(x):
     return(np.round(1.0*x * (1<<31)))

def q15accuracy(x):
     return(np.round(1.0*x * (1<<15)))

def q7accuracy(x):
     return(np.round(1.0*x * (1<<7)))

def Q31toF32(x):
     return(1.0*x / 2**31)

def Q15toF32(x):
     return(1.0*x / 2**15)

def Q7toF32(x):
     return(1.0*x / 2**7)

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

# For benchmarks
NBSAMPLES=256

def cartesian(*somelists):
   r=[]
   for element in itertools.product(*somelists):
       r.append(element)
   return(r)

# Fixed point division should not be called with a denominator of zero.
# But if it is, it should return a saturated result.
def divide(f,r):
    e = 0
    a,b=r

    if f == Tools.Q31:
        e = 1.0 / (1<<31)
        a = 1.0*q31accuracy(a) / (2**31)
        b = 1.0*q31accuracy(b) / (2**31)
    if f == Tools.Q15:
        e = 1.0 / (1<<15)
        a = 1.0*q15accuracy(a) / (2**15)
        b = 1.0*q15accuracy(b) / (2**15)
    if f == Tools.Q7:
        e = 1.0 / (1<<7)
        a = 1.0*q7accuracy(a) / (2**7)
        b = 1.0*q7accuracy(b) / (2**7)

    if b == 0.0:
        if a >= 0.0:
           return(1.0,0)
        else:
           return(-1.0,0)
        
    k = 0
    while abs(a) > abs(b):
       a = a / 2.0
       k = k + 1 
    # In C code we don't saturate but instead generate the right value
    # with a shift of 1.
    # So this test is to ease the comparison between the Python reference
    # and the output of the division algorithm in C
    if abs(a/b) > 1 - e:
       a = a / 2.0
       k = k + 1 

    return(a/b,k)


def initLogValues(format):
    if format == Tools.Q15:
        vals=np.linspace(np.float_power(2,-15),1.0,num=125)
    elif format == Tools.F16:
        vals=np.linspace(np.float_power(2,-10),1.0,num=125)
    else:
        vals=np.linspace(np.float_power(2,-31),1.0,num=125)


    ref=np.log(vals)
    if format==Tools.Q31 :
        # Format must be Q5.26
        ref = ref / 32.0
    if format == Tools.Q15:
        # Format must be Q4.11
        ref = ref / 16.0
    return(vals,ref)

def normalizeToOne(x):
    s = 0
    while (abs(x)>1):
        x = x /2.0 
        s = s + 1 
    return(int(s),x)

def writeTests(config,format):
    
    a1=np.array([0,math.pi/4,math.pi/2,3*math.pi/4,math.pi,5*math.pi/4,3*math.pi/2,2*math.pi-1e-6])
    a2=np.array([-math.pi/4,-math.pi/2,-3*math.pi/4,-math.pi,-5*math.pi/4,-3*math.pi/2,-2*math.pi-1e-6])
    a3 = a1 + 2*math.pi  
    angles=np.concatenate((a1,a2,a3))
    refcos = np.cos(angles)
    refsin = np.sin(angles)


    vals=np.linspace(0.0,1.0,1024)
    sqrtvals=np.sqrt(vals)

    # Negative values in CMSIS are giving 0
    vals[0] = -0.4
    sqrtvals[0] = 0.0
    
    if format != Tools.F64 and format != 0 and format != 16:
        angles=np.concatenate((a1,a2,a1))
        angles = angles / (2*math.pi)
    config.writeInput(1, angles,"Angles")
    
    config.writeInput(1, vals,"SqrtInput")
    config.writeReference(1, sqrtvals,"Sqrt")

    config.writeReference(1, refcos,"Cos")
    config.writeReference(1, refsin,"Sin")


    # For benchmarks
    samples=np.random.randn(NBSAMPLES)
    samples = np.abs(Tools.normalize(samples))
    config.writeInput(1, samples,"Samples")

    numerator=np.linspace(-0.9,0.9)
    numerator=np.hstack([numerator,np.array([-1.0,1.0])])
    denominator=np.linspace(-0.9,0.9)
    denominator=np.hstack([denominator,np.array([-1.0,1.0])])

    samples=cartesian(numerator,denominator)
    numerator=[x[0] for x in samples]
    denominator=[x[1] for x in samples]

    result=[divide(format,x) for x in samples]

    resultValue=[x[0] for x in result]
    resultShift=[x[1] for x in result]

    config.setOverwrite(False)
    config.writeInput(1, numerator,"Numerator")
    config.writeInput(1, denominator,"Denominator")
    config.writeReference(1, resultValue,"DivisionValue")
    config.writeReferenceS16(1, resultShift,"DivisionShift")
    config.setOverwrite(False)


    vals,ref=initLogValues(format)
    config.writeInput(1, vals,"LogInput")
    config.writeReference(1, ref,"Log")

    config.setOverwrite(False)

    # Testing of ATAN2
    angles=np.linspace(0.0,2*math.pi,1000,endpoint=True)
    angles=np.hstack([angles,np.array([math.pi/4.0])])
    if format == Tools.Q31 or format == Tools.Q15:
        radius=[1.0]
    else:
        radius=np.linspace(0.1,0.9,10,endpoint=True)
    combinations = cartesian(radius,angles)
    res=[]
    yx = []
    for r,angle in combinations:
        x = r*np.cos(angle)
        y = r*np.sin(angle)
        res.append(np.arctan2(y,x))
        yx.append(y)
        yx.append(x)

    
    config.writeInput(1, np.array(yx).flatten(),"Atan2Input")

    # Q2.29 or Q2.13 to represent PI in the output
    if format == Tools.Q31 or format == Tools.Q15:
       config.writeReference(1, np.array(res)/4.0,"Atan2Ref")
    else:
       config.writeReference(1, np.array(res),"Atan2Ref")

    config.setOverwrite(False)

    if format == Tools.Q31 or format == Tools.Q15:

       if format == Tools.Q31:
          theInput=np.array([1.0-1e-6,0.6,0.5,0.3,0.25,0.1,1.0/(1<<31)])

       if format == Tools.Q15:
          theInput=np.array([1.0-1e-6,0.6,0.5,0.3,0.25,0.1,1.0/(1<<15)])

       ref=1.0 / theInput
       shiftAndScaled=np.array([normalizeToOne(x) for x in ref]).transpose()
       shiftValues=shiftAndScaled[0].astype(np.int16)
       scaledValues=shiftAndScaled[1]
       #print(shiftAndScaled)


       config.writeInput(1, np.array(theInput),"RecipInput")
       config.writeReference(1, scaledValues,"RecipRef")
       config.writeReferenceS16(1, shiftValues,"RecipShift")

       

def tocint32(x):
    if x < 0:
        return((0x10000000000000000 + x) & 0xFFFFFFFF)
    else:
        return(x & 0xFFFFFFFF)

# C and Python are not rounding the integer division
# in the same way
def cdiv(a,b):
    sign = 1
    if ((a<0) and (b>0)) or ((a>0) and (b<0)):
        sign = -1 

    a= abs(a)
    b = abs(b)

    d = sign*(a // b)

    return(d)

def testInt64(config):
    theInput=[0x1000000080000000,
                 0x0000000080000000,
                 0x0000000020000000,
                 0x0000000000000000]

    ref=[0x40000002,
         0x40000000,
         0x40000000,
         0
    ] 
    norms=[-30,-1,1,0]
    config.writeInputU64(1,np.array(theInput),"Norm64To32_Input")
    config.writeReferenceS16(1,norms,"RefNorm64To32_Norms")
    config.writeReferenceS32(1,ref,"RefNorm64To32_Vals")

    config.setOverwrite(False)
    
    allCombinations=[(0x7FFFFFFFFFFFFFFF,2),
    (-0x7FFFFFFFFFFFFFFF-1,2),
    ( 0x4000000000000000,0x7FFFFFFF),
    ( -0x4000000000000000,0x7FFFFFFF),
    (  0x2000000000000000,0x7FFFFFFF),
    ( -0x2000000000000000,0x7FFFFFFF),
    (  0x1000000000000000,0x7FFFFFFF),
    ( -0x1000000000000000,0x7FFFFFFF),
    (  0x0000000080000000,2),
    ( -0x0000000080000000,2),
    (  0x0000000040000000,2),
    ( -0x0000000080000000,2)
    ]

    res = [tocint32(cdiv(x,y))  for (x,y) in allCombinations]
    
    allCombinations=np.array(allCombinations,dtype=np.int64).flatten()
    config.writeInputS64(1,allCombinations[0::2],"DivDenInput")
    config.writeInputS32(1,allCombinations[1::2],"DivNumInput")

    config.writeReferenceU32(1, res,"DivRef")
    config.setOverwrite(False)


def writeTestsFloat(config,format):

    writeTests(config,format)

    data1 = np.random.randn(20)
    data1 = np.abs(data1)
    data1 = data1 + 1e-3 # To avoid zero values
    data1 = Tools.normalize(data1)


    samples=np.concatenate((np.array([0.0,1.0]),np.linspace(-0.4,0.4)))
    config.writeInput(1, samples,"ExpInput")
    v = np.exp(samples)
    config.writeReference(1, v,"Exp")

    # For benchmarks and other tests
    samples=np.random.randn(NBSAMPLES)
    samples = np.abs(Tools.normalize(samples))
    config.writeInput(1, samples,"Samples")

    v = 1.0 / samples
    config.writeReference(1, v,"Inverse")




    
def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","FastMath","FastMath")
    PARAMDIR = os.path.join("Parameters","DSP","FastMath","FastMath")
    
    configf64=Tools.Config(PATTERNDIR,PARAMDIR,"f64")
    configf32=Tools.Config(PATTERNDIR,PARAMDIR,"f32")
    configf16=Tools.Config(PATTERNDIR,PARAMDIR,"f16")
    configq31=Tools.Config(PATTERNDIR,PARAMDIR,"q31")
    configq15=Tools.Config(PATTERNDIR,PARAMDIR,"q15")

    configq64=Tools.Config(PATTERNDIR,PARAMDIR,"q63")

    

    configf64.setOverwrite(False)
    configf32.setOverwrite(False)
    configf16.setOverwrite(False)
    configq31.setOverwrite(False)
    configq15.setOverwrite(False)
    configq64.setOverwrite(False)

    writeTestsFloat(configf64,Tools.F64)
    writeTestsFloat(configf32,0)
    writeTestsFloat(configf16,16)
    writeTests(configq31,31)
    writeTests(configq15,15)

    testInt64(configq64)


if __name__ == '__main__':
  generatePatterns()

