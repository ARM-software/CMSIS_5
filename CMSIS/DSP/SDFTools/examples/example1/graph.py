from cmsisdsp.sdf.scheduler import *

### Define new types of Nodes 

class Node(GenericNode):
    def __init__(self,name,theType,inLength,outLength):
        GenericNode.__init__(self,name)
        self.addInput("i",theType,inLength)
        self.addOutput("o",theType,outLength)

class Sink(GenericSink):
    def __init__(self,name,theType,inLength):
        GenericSink.__init__(self,name)
        self.addInput("i",theType,inLength)

    @property
    def typeName(self):
        return "Sink"

class Source(GenericSource):
    def __init__(self,name,theType,inLength):
        GenericSource.__init__(self,name)
        self.addOutput("o",theType,inLength)

    @property
    def typeName(self):
        return "Source"

class ProcessingNode(Node):
    @property
    def typeName(self):
        return "ProcessingNode"



### Define nodes
floatType=CType(F32)
src=Source("source",floatType,5)
b=ProcessingNode("filter",floatType,7,5)
b.addLiteralArg(4)
b.addLiteralArg("Test")
b.addVariableArg("someVariable")
sink=Sink("sink",floatType,5)

g = Graph()

g.connect(src.o,b.i)
g.connect(b.o,sink.i)


print("Generate graphviz and code")

conf=Configuration()
conf.debugLimit=1
conf.cOptionalArgs="int someVariable"
#conf.displayFIFOSizes=True
# Prefix for global FIFO buffers
#conf.prefix="sched1"

#print(g.nullVector())
sched = g.computeSchedule()
#print(sched.schedule)
print("Schedule length = %d" % sched.scheduleLength)
print("Memory usage %d bytes" % sched.memory)
#

#conf.codeArray=True
sched.ccode("generated",conf)

with open("test.dot","w") as f:
    sched.graphviz(f)

