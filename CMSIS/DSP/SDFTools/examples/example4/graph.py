import numpy as np 

from cmsisdsp.sdf.scheduler import *


FS=16000
# You can try with 120
AUDIO_INTERRUPT_LENGTH = 192
WINSIZE=256
OVERLAP=128
floatType=CType(F32)


### Define nodes
src=FileSource("src",AUDIO_INTERRUPT_LENGTH)
src.addLiteralArg("input_example3.txt")

sliding=SlidingBuffer("audioWin",floatType,WINSIZE,OVERLAP)
overlap=OverlapAdd("audioOverlap",floatType,WINSIZE,OVERLAP)
window=Dsp("mult",floatType,WINSIZE)

toCmplx=ToComplex("toCmplx",floatType,WINSIZE)
toReal=ToReal("toReal",floatType,WINSIZE)
fft=CFFT("cfft",floatType,WINSIZE)
ifft=ICFFT("icfft",floatType,WINSIZE)

hann=Constant("HANN")
sink=FileSink("sink",AUDIO_INTERRUPT_LENGTH)
sink.addLiteralArg("output_example3.txt")
sink.addVariableArg("dispbuf")

g = Graph()

g.connect(src.o, sliding.i)

# Windowing
g.connect(sliding.o, window.ia)
g.connect(hann,window.ib)

# FFT
g.connect(window.o,toCmplx.i)
g.connect(toCmplx.o,fft.i)
g.connect(fft.o,ifft.i)
g.connect(ifft.o,toReal.i)


# Overlap add
g.connect(toReal.o,overlap.i)
g.connect(overlap.o,sink.i)


print("Generate graphviz and code")



#print(g.nullVector())
sched = g.computeSchedule()
#print(sched.schedule)
print("Schedule length = %d" % sched.scheduleLength)
print("Memory usage %d bytes" % sched.memory)
#
conf=Configuration()
conf.debugLimit=42
conf.pyOptionalArgs="dispbuf"
#conf.dumpFIFO=True
#conf.prefix="sched1"
sched.pythoncode(".",config=conf)

with open("test.dot","w") as f:
    sched.graphviz(f)

