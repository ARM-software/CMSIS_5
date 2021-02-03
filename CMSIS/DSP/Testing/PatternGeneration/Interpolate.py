import os.path
import numpy as np
import itertools
import Tools
from scipy.interpolate import interp1d,interp2d,CubicSpline

# Those patterns are used for tests and benchmarks.
# For tests, there is the need to add tests for saturation

# Get lists of points in row order for use in CMSIS function
def getLinearPoints(x,y):
     return(np.array([[p[1],p[0]] for p in np.array(np.meshgrid(y,x)).T.reshape(-1,2)]))

def writeTests(config,format):
    # Linear interpolation test
    NBSAMPLES=40

    x = np.linspace(0, NBSAMPLES, num=NBSAMPLES+1, endpoint=True)
    y = np.cos(-x**2/(NBSAMPLES - 1))
    f = interp1d(x, y)
    data=x+0.5
    data=data[:-1]
    z = f(data)

    if format != 0 and format != 16:
       data = data / 2.0**11
    if format != 0 and format != 16:
       config.writeInputQ31(1, data,"Input")
    else:
       config.writeInput(1, data)
    config.writeInput(1, y,"YVals")
    
    ref = z
    config.writeReference(1, ref)

    # Bilinear interpolation test
    x = np.arange(-3.14, 3.14, 1.0)
    y = np.arange(-3.14, 3.14, 0.8)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+yy**2)
    f = interp2d(x, y, z, kind='linear')


    # Configuration for the test (to initialize the bilinear structure)
    matrixSize=[np.size(x),np.size(y)]

    # Generate reference value for bilinear instance
    # getLinearPoints ensure they are in row order
    samples = getLinearPoints(x,y)
    # We recompute the value of the function on the samples in row
    # order
    yvals = np.array([np.sin(i[0]**2+i[1]**2) for i in samples])


    # Now we generate other points. The points where we want to evaluate
    # the function.
    # In Python they must be rescale between -3.14 and tghe max x or max y defined above.
    # In CMSIS they will be between 1 and numRow-1 or numCols-1.
    # Since we add 0.5 to be sure we are between grid point, we use
    # numCols-2 as bound to be sured we are <= numCols-1
    numCols = np.size(x)
    numRows = np.size(y)

    NBX = 10
    NBY = 15

    # The CMSIS indexes
    ix = np.linspace(0, numCols-3, num=NBX, endpoint=True)+0.5
    iy = np.linspace(0, numRows-3, num=NBY, endpoint=True)+0.5


    # The corresponding Python values
    ixVal = ((ix ) / (numCols-1)) * (x[-1] + 3.14) - 3.14
    iyVal = ((iy ) / (numRows-1)) * (y[-1] + 3.14) - 3.14
    
    # Input samples for CMSIS.
    inputSamples = getLinearPoints(ix,iy)
    
    # We compute the Python interpolated function on the values
    inputVals = getLinearPoints(ixVal,iyVal)
    ref=np.array([f(i[0],i[1]) for i in inputVals])


    if format != 0 and format != 16:
       inputSamples = inputSamples / 2.0**11
    data = inputSamples.reshape(np.size(inputSamples))
    if format != 0 and format != 16:
       config.writeInputQ31(2, data,"Input")
    else:
       config.writeInput(2, data)

    config.writeInput(2, yvals.reshape(np.size(yvals)),"YVals")
    config.writeReference(2, ref.reshape(np.size(ref)))
    config.writeInputS16(2, matrixSize,"Config")


    
    x = [0,3,10,20]
    config.writeInput(3,x,"InputX")
    y = [0,9,100,400]
    config.writeInput(3,y,"InputY")
    xnew = np.arange(0,20,1)
    config.writeInput(3,xnew,"OutputX")
    ynew = CubicSpline(x,y)
    config.writeReference(3, ynew(xnew))

    x = np.arange(0, 2*np.pi+np.pi/4, np.pi/4)
    config.writeInput(4,x,"InputX")
    y = np.sin(x)
    config.writeInput(4,y,"InputY")
    xnew = np.arange(0, 2*np.pi+np.pi/16, np.pi/16)
    config.writeInput(4,xnew,"OutputX")
    ynew = CubicSpline(x,y,bc_type="natural")
    config.writeReference(4, ynew(xnew))

    x = [0,3,10]
    config.writeInput(5,x,"InputX")
    y = x
    config.writeInput(5,y,"InputY")
    xnew = np.arange(-10,20,1)
    config.writeInput(5,xnew,"OutputX")
    ynew = CubicSpline(x,y)
    config.writeReference(5, ynew(xnew))




def generatePatterns():
    PATTERNDIR = os.path.join("Patterns","DSP","Interpolation","Interpolation")
    PARAMDIR = os.path.join("Parameters","DSP","Interpolation","Interpolation")
    
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
