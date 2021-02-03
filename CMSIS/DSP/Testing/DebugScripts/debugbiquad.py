import numpy as np
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,semilogx, semilogy
import scipy.fftpack
import os.path 
import struct
import argparse
import sys
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

import PatternGeneration.DebugTools as d

f = "f16"

inputPath = os.path.join(parent_dir,"Patterns","DSP","Filtering","BIQUAD","BIQUAD%s" % f.upper(),"BiquadInput1_%s.txt" % f )
refPath = os.path.join(parent_dir,"Patterns","DSP","Filtering","BIQUAD","BIQUAD%s" % f.upper(),"BiquadOutput1_%s.txt" % f)
outputPath= os.path.join(parent_dir,"Output","DSP","Filtering","BIQUAD","BIQUAD%s" % f.upper(),"Output_1.txt")




inSig = d.readF16Pattern(inputPath)
     
refSig = d.readF16Pattern(refPath)
     
sig = d.readF16Output(outputPath)


figure()
plot(inSig)
figure()
plot(refSig)
figure()
plot(sig)

#print(d.SNR(refSig,sig))

#figure()
#plot(np.unwrap(np.angle(refSig)))
#figure()
#plot(np.unwrap(np.angle(sig)))
#figure()
#plot(np.unwrap(np.angle(sig)) - np.unwrap(np.angle(refSig)))
show()#