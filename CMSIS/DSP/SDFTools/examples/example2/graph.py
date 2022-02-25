from cmsisdsp.sdf.scheduler import *


AUDIO_INTERRUPT_LENGTH = 160
MFCCFEATURESSIZE=10
NBOUTMFCC=50
OVERLAPMFCC=25
floatType=CType(F32)


### Define new types of Nodes 

class Node(GenericNode):
    def __init__(self,name,theType,inLength,outLength):
        GenericNode.__init__(self,name)
        self.addInput("i",theType,inLength)
        self.addOutput("o",theType,outLength)

    @property
    def typeName(self):
        return "Node"

class StereoSource(GenericSource):
    def __init__(self,name,inLength):
        GenericSource.__init__(self,name)
        self.addOutput("o",floatType,2*inLength)

    @property
    def typeName(self):
        return "StereoSource"

# This is a fake TFLite node just for illustration
class TFLite(GenericSink):
    def __init__(self,name):
        GenericSink.__init__(self,name)
        self.addInput("i",floatType,NBOUTMFCC*MFCCFEATURESSIZE)

    @property
    def typeName(self):
        return "TFLite"

# This is a fake MFCC just to illustrate how it could be used in a graph.
# For a real MFCC example, look at example5
class MFCC(GenericNode):
    def __init__(self,name,inLength):
        GenericNode.__init__(self,name)

        self.addInput("i",floatType,inLength)
        self.addOutput("o",floatType,MFCCFEATURESSIZE)

    @property
    def typeName(self):
        return "MFCC"


### Define nodes
half=Constant("HALF")
src=StereoSource("src",AUDIO_INTERRUPT_LENGTH)
toMono=Unzip("toMono",floatType, AUDIO_INTERRUPT_LENGTH)

sa=Dsp("scale",floatType,AUDIO_INTERRUPT_LENGTH)
sb=Dsp("scale",floatType,AUDIO_INTERRUPT_LENGTH)

add=Dsp("add",floatType,AUDIO_INTERRUPT_LENGTH)

audioWindow=SlidingBuffer("audioWin",floatType,640,320)
mfcc=MFCC("mfcc",640)

mfccWindow=SlidingBuffer("mfccWind",floatType,NBOUTMFCC*MFCCFEATURESSIZE,OVERLAPMFCC*MFCCFEATURESSIZE)
tf=TFLite("TFLite")

g = Graph()

g.connectWithDelay(src.o, toMono.i,10)
g.connect(toMono.o1,sa.ia)
g.connect(toMono.o2,sb.ia)

# A constant node as no output
# So it is connected directly
g.connect(half,sa.ib)
g.connect(half,sb.ib)

g.connect(sa.o,add.ia)
g.connect(sb.o,add.ib)

g.connect(add.o,audioWindow.i)
g.connect(audioWindow.o, mfcc.i)
g.connect(mfcc.o,mfccWindow.i)
g.connect(mfccWindow.o,tf.i)

print("Generate graphviz and code")

conf=Configuration()
conf.debugLimit=1
conf.cOptionalArgs="int opt1,int opt2"
#conf.memoryOptimization=True
#print(g.nullVector())
sched = g.computeSchedule(conf)
#print(sched.schedule)
print("Schedule length = %d" % sched.scheduleLength)
print("Memory usage %d bytes" % sched.memory)
#

#conf.codeArray=True
sched.ccode("generated",conf)

with open("test.dot","w") as f:
    sched.graphviz(f)

