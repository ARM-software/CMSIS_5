import numpy as np
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,semilogx, semilogy
import scipy.fftpack
import os.path 
import struct
import argparse

import PatternGeneration.DebugTools as d

# Example script to read patterns and test outputs

parser = argparse.ArgumentParser(description='Debug description')
parser.add_argument('-f', nargs='?',type = str, default="f32", help="Format")
parser.add_argument('-n', nargs='?',type = str, default="1", help="Test number")
parser.add_argument('-i', nargs='?',type = bool, default=False, help="Ifft")
parser.add_argument('-ui', nargs='?',const=True,type = bool, default=False, help="Display curves")

args = parser.parse_args()

FFTSIZES=[16,32,64,128,256,512,1024,2048,4096]

if int(args.n) >= 19:
    args.i = True 

if args.i:
   n = int(args.n) - 18
   s = FFTSIZES[n-1]
   sc = n - 1 + 4
   inputPath = os.path.join("Patterns","DSP","Transform","Transform%s" % args.f.upper(),"ComplexInputIFFTSamples_Noisy_%d_%d_%s.txt" % (s,n,args.f))
   refPath = os.path.join("Patterns","DSP","Transform","Transform%s" % args.f.upper(),"ComplexInputSamples_Noisy_%d_%d_%s.txt" % (s,n,args.f))
   outputPath= os.path.join("Output","DSP","Transform","Transform%s" % args.f.upper(),"ComplexFFTSamples_%s.txt" % args.n)
else:
   s = FFTSIZES[int(args.n)-1]
   inputPath = os.path.join("Patterns","DSP","Transform","Transform%s" % args.f.upper(),"ComplexInputSamples_Noisy_%d_%s_%s.txt" % (s,args.n,args.f))
   refPath = os.path.join("Patterns","DSP","Transform","Transform%s" % args.f.upper(),"ComplexFFTSamples_Noisy_%d_%s_%s.txt" % (s,args.n,args.f))
   outputPath= os.path.join("Output","DSP","Transform","Transform%s" % args.f.upper(),"ComplexFFTSamples_%s.txt" % args.n)

print(inputPath)


if args.f == "f32":
    inSig = d.readF32Pattern(inputPath)
    inSig=inSig.view(dtype=np.complex128)
     
    refSig = d.readF32Pattern(refPath)
    refSig=refSig.view(dtype=np.complex128)
     
    sig = d.readF32Output(outputPath)
    sig=sig.view(dtype=np.complex128)

if args.f == "q31":
    inSig = d.readQ31Pattern(inputPath)
    inSig=inSig.view(dtype=np.complex128)
     
    refSig = d.readQ31Pattern(refPath)
    refSig=refSig.view(dtype=np.complex128)
     
    sig = d.readQ31Output(outputPath)
    sig=sig.view(dtype=np.complex128)

if args.f == "q15":
    inSig = d.readQ15Pattern(inputPath)
    inSig=inSig.view(dtype=np.complex128)
     
    refSig = d.readQ15Pattern(refPath)
    refSig=refSig.view(dtype=np.complex128)
     
    sig = d.readQ15Output(outputPath)
    sig=sig.view(dtype=np.complex128)


if args.i and args.f != "f32":
    refSig = refSig / 2**sc

if args.ui:
   if args.i:
      figure()
      plot(abs(inSig))
      figure()
      plot(np.real(refSig))
      figure()
      plot(np.real(sig))
   else:
      figure()
      plot(np.real(inSig))
      figure()
      plot(abs(refSig))
      figure()
      plot(abs(sig))

print(d.SNR(refSig,sig))

#figure()
#plot(np.unwrap(np.angle(refSig)))
#figure()
#plot(np.unwrap(np.angle(sig)))
#figure()
#plot(np.unwrap(np.angle(sig)) - np.unwrap(np.angle(refSig)))
show()#