import numpy as np 

from cmsisdsp.sdf.scheduler import *


from sharedconfig import *



floatType = CType(F32)

### The feature computed by the graph is one second of MFCCs.
### So the nbMFCCOutputs is computed from this and with the additional
### assumption that it must be even.
### Because the MFCC slising window is sliding by half a second
### each time (half number of MFCCs)
src=FileSource("src",NBCHANNELS*AUDIO_INTERRUPT_LENGTH)
src.addLiteralArg("input_example6.txt")


slidingAudio=SlidingBuffer("audioWin",floatType,FFTSize,FFTSize>>1)
slidingMFCC=SlidingBuffer("mfccWin",floatType,2*numOfDctOutputs,numOfDctOutputs)

mfcc=MFCC("mfcc",floatType,FFTSize,numOfDctOutputs)
mfcc.addVariableArg("mfccConfig")

sink=FileSink("sink",numOfDctOutputs)
sink.addLiteralArg("output_example6.txt")

g = Graph()

g.connect(src.o, slidingAudio.i)

g.connect(slidingAudio.o, mfcc.i)

g.connect(mfcc.o,slidingMFCC.i)
g.connect(slidingMFCC.o,sink.i)

print("Generate graphviz and code")



sched = g.computeSchedule()
print("Schedule length = %d" % sched.scheduleLength)
print("Memory usage %d bytes" % sched.memory)
#
conf=Configuration()
conf.debugLimit=1
conf.cOptionalArgs="arm_mfcc_instance_f32 *mfccConfig"

#conf.codeArray=True
sched.ccode("generated",config=conf)

with open("test.dot","w") as f:
    sched.graphviz(f)

