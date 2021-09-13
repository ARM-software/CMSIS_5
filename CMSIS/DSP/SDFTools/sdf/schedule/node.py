###########################################
# Project:      CMSIS DSP Library
# Title:        node.py
# Description:  Node class for description of dataflow graph
# 
# $Date:        29 July 2021
# $Revision:    V1.10.0
# 
# Target Processor: Cortex-M and Cortex-A cores
# -------------------------------------------------------------------- */
# 
# Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
# 
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
############################################
"""Description of the basic types used to build a dataflow graph"""
from jinja2 import Environment, PackageLoader, select_autoescape

def camelCase(st):
    output = ''.join(x for x in st.title() if x.isalnum())
    return output[0].lower() + output[1:]

def joinit(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

### Definition of the IOs

class IO:
    """Class of input / outputs"""
    def __init__(self,owner,name,theType,nbSamples):
        self._theType = theType
        self._nbSamples = nbSamples 
        self._owner = owner
        self._name = name
        self._fifo = None 
        self.constantNode = None

    @property
    def fifo(self):
        return self._fifo

    ## the attribute name and the method name must be same which is used to set the value for the attribute
    @fifo.setter
    def fifo(self, var):
        self._fifo = var

   

    def compatible(self,other):
        return(self._theType == other._theType)

    @property
    def owner(self):
        return self._owner

    @property
    def name(self):
        return self._name

    @property
    def ctype(self):
        """ctype string """
        return self._theType.ctype

    @property
    def nptype(self):
        """ctype string """
        return self._theType.nptype

    @property
    def theType(self):
        return self._theType

    @property
    def dspExtension(self):
        return self._theType.dspExtension

    @property
    def graphViztype(self):
        return self._theType.graphViztype

    @property
    def nbSamples(self):
        return self._nbSamples
    
    

class Input(IO):
     """Node input"""
     pass

class Output(IO):
     """Node output"""
     pass

### Definition of the nodes types

class Constant:
    """ Represent a constant object.

        A constant object is ignored for the scheduling.
        But it can be connected to CMSIS-DSP inputs.
        It is generated as DEFINE
    """
    def __init__(self,name):
        self._name = name 

    @property
    def name(self):
        return self._name

    @property
    def isConstantNode(self):
        return True
    
    


class BaseNode:
    """Root class for all Nodes of a dataflow graph.
       To define a new kind of node, inherit from this class"""

    def __init__(self,name):
        """Create a new kind of Node.

        name :: The name of the node which is used as
        a C variable in final code"""
        self._nodeName = name
        self._nodeID = name
        self._inputs={}
        self._outputs={}
        # For code generations
        # The fifo args
        self._args=""
        # Literal arguments
        self.literalArgs=None 
        self.variableArguments=None

    def __getattr__(self,name):
        """Present inputs / outputs as attributes"""
        if name in self._inputs:
           return(self._inputs[name])
        if name in self._outputs:
           return(self._outputs[name])
        raise AttributeError

    def __getitem__(self,name):
        """Present inputs / outputs as keys"""
        if name in self._inputs:
           return(self._inputs[name])
        if name in self._outputs:
           return(self._outputs[name])
        raise IndexError 

    def addLiteralArg(self,l):
        if self.literalArgs:
            self.literalArgs.append(l)
        else:
            self.literalArgs=[l]

    def addVariableArg(self,l):
        if self.variableArguments:
            self.variableArguments.append(l)
        else:
            self.variableArguments=[l]

    @property
    def isConstantNode(self):
        return False

    @property
    def hasState(self):
        """False if the node is a pure functiom with no state
           and no associated C++ object
        """
        return(True)

    @property
    def typeName(self):
        return "void"
    
    
    @property
    def nodeID(self):
        """Node ID to uniquely identify a node"""
        return self._nodeID

    @property
    def nodeName(self):
        """Node name displayed in graph

           It could be the same for different nodes if the
           node is just a function with no state.
        """
        return self._nodeName

    # For code generation



    def allIOs(self):
        """Get list of IO objects for inputs and outputs"""
        ins=[] 
        outs=[]
        # Use orderd io names
        for io in self.inputNames:
            x = self._inputs[io]
            ins.append(x)

        for io in self.outputNames:
            x = self._outputs[io]
            outs.append(x)

        
        return(ins,outs)

    def ioTemplate(self):
        """Template arguments for C
           input type, input size ...
           output type, output size ...

           Some nodes may customize it
        """
        ios=[] 
        # Use ordered io names
        for io in self.inputNames:
            x = self._inputs[io]
            ios.append("%s,%d" % (x.ctype,x.nbSamples))

        for io in self.outputNames:
            x = self._outputs[io]
            ios.append("%s,%d" % (x.ctype,x.nbSamples))

        
        return("".join(joinit(ios,",")))

    def pythonIoTemplate(self):
        """Template arguments for Python
           input type, input size ...
           output type, output size ...

           Some nodes may customize it
        """
        ios=[] 
        # Use ordered io names
        for io in self.inputNames:
            x = self._inputs[io]
            ios.append("%d" % x.nbSamples)

        for io in self.outputNames:
            x = self._outputs[io]
            ios.append("%d" % x.nbSamples)

        
        return("".join(joinit(ios,",")))

    def cRun(self,ctemplate=True):
        """Run function

           Some nodes may customize it
        """
        if ctemplate:
           return ("sdfError = %s.run();" % self.nodeName)
        else:
           return ("sdfError = %s.run()" % self.nodeName)
    
    
    @property
    def listOfargs(self):
        """List of fifos args for object initialization"""
        return self._args

    @property
    def args(self):
        """String of fifo args for object initialization
            with literal argument and variable arguments"""
        allArgs=self.listOfargs
        if self.literalArgs:
            for lit in self.literalArgs:
                if isinstance(lit,str):
                    allArgs.append("\"%s\"" % lit) 
                else:
                    allArgs.append(str(lit))
        if self.variableArguments:
            allArgs += self.variableArguments
        return "".join(joinit(allArgs,","))
    
    @args.setter
    def args(self,fifoIDs):
       res=[]
       for x in fifoIDs:
         # If args is a FIFO we generate a name using fifo ids
         if isinstance(x,int):
            res.append("fifo%d" % x)
         # If args is a constant node, we just use the constant node name
         # (Defined in C code)
         else:
            res.append(x)
       self._args=res

    # For graphviz generation



    @property
    def graphvizName(self):
        """Name for graph vizualization"""
        return ("%s<BR/>(%s)" % (self.nodeName,self.typeName))

    @property
    def inputNames(self):
        return sorted(list(self._inputs.keys()))


    @property
    def outputNames(self):
        return sorted(list(self._outputs.keys()))

    @property
    def hasManyInputs(self):
        return len(self._inputs.keys())>1

    @property
    def hasManyOutputs(self):
        return len(self._outputs.keys())>1

    @property
    def hasManyIOs(self):
        return (self.hasManyInputs or self.hasManyOutputs)

    @property
    def nbEmptyInputs(self):
        return (self.maxNbIOs - len(self._inputs.keys()))
    
    @property
    def nbEmptyOutputs(self):
        return (self.maxNbIOs - len(self._outputs.keys()))

    @property
    def maxNbIOs(self):
        return max(len(self._inputs.keys()),len(self._outputs.keys()))
    
    

class GenericSink(BaseNode):
    """A sink in the dataflow graph""" 

    def __init__(self,name):
        BaseNode.__init__(self,name)

    @property
    def typeName(self):
        return "void"

    def addInput(self,name,theType,theLength):
        self._inputs[name]=Input(self,name,theType,theLength)


class GenericSource(BaseNode):
    """A source in the dataflow graph""" 

    def __init__(self,name):
        BaseNode.__init__(self,name)

    @property
    def typeName(self):
        return "void"

    def addOutput(self,name,theType,theLength):
        self._outputs[name]=Output(self,name,theType,theLength)

class GenericNode(BaseNode):
    """A source in the dataflow graph""" 

    def __init__(self,name):
        BaseNode.__init__(self,name)
    
    @property
    def typeName(self):
        return "void"

    def addInput(self,name,theType,theLength):
        self._inputs[name]=Input(self,name,theType,theLength)

    def addOutput(self,name,theType,theLength):
        self._outputs[name]=Output(self,name,theType,theLength)

class SlidingBuffer(GenericNode):

    def __init__(self,name,theType,length,overlap):
        GenericNode.__init__(self,name)
        self._length = length 
        self._overlap = overlap 
        self.addInput("i",theType,length-overlap)
        self.addOutput("o",theType,length)
    
    def ioTemplate(self):
        """ioTemplate is different for window
        """
        theType=self._inputs[self.inputNames[0]].ctype  
        ios="%s,%d,%d" % (theType,self._length,self._overlap)
        return(ios)

    def pythonIoTemplate(self):
        """ioTemplate is different for window
        """
        theType=self._inputs[self.inputNames[0]].ctype  
        ios="%d,%d" % (self._length,self._overlap)
        return(ios)
        

    @property
    def typeName(self):
        return "SlidingBuffer"

class OverlapAdd(GenericNode):

    def __init__(self,name,theType,length,overlap):
        GenericNode.__init__(self,name)
        self._length = length 
        self._overlap = overlap 
        self.addInput("i",theType,length)
        self.addOutput("o",theType,overlap)
    
    def ioTemplate(self):
        """ioTemplate is different for window
        """
        theType=self._inputs[self.inputNames[0]].ctype  
        ios="%s,%d,%d" % (theType,self._length,self._overlap)
        return(ios)

    def pythonIoTemplate(self):
        """ioTemplate is different for window
        """
        theType=self._inputs[self.inputNames[0]].ctype  
        ios="%d,%d" % (self._length,self._overlap)
        return(ios)

    
        

    @property
    def typeName(self):
        return "OverlapAdd"




# Pure compute functions
# It is supporting unary function (src,dst,blockize)
# and binary functions (sraa,srcb, dst, blocksize)
# For cmsis, the prefix arm and the type suffix are not needed
# if class Dsp is used
class GenericFunction(GenericNode):
    # Number of function node of each category
    # Used to generate unique ID and names when
    # unique names are required
    # like for creating the graph where each call to
    # the same functiion must be identified as a
    # separate node
    NODEID={}

    ENV = Environment(
       loader=PackageLoader("sdf"),
       autoescape=select_autoescape(),
       lstrip_blocks=True,
       trim_blocks=True
    )
    
    CTEMPLATE = ENV.get_template("cmsis.cpp")
    PYTEMPLATE = ENV.get_template("cmsis.py")

    def __init__(self,funcname,theType,length):
        if not (funcname in Dsp.NODEID):
            Dsp.NODEID[funcname]=1 

        GenericNode.__init__(self,"%s%d" % (funcname,Dsp.NODEID[funcname]))

        self._hasState = False
        self._length = length 
        self._nodeName = funcname

        Dsp.NODEID[funcname]=Dsp.NODEID[funcname]+1


    @property
    def hasState(self):
        return self._hasState
    

    @property
    def typeName(self):
        return "Function"

    # To clean
    def cRun(self,ctemplate=True):
       if ctemplate:
         theType=self._inputs[self.inputNames[0]].ctype
       else:
         theType=self._inputs[self.inputNames[0]].nptype
       theLen = self._inputs[self.inputNames[0]].nbSamples
       theId = 0
       # List of buffer and corresponding fifo to initialize buffers
       inputs=[]
       outputs=[]
       # List of buffers variable to declare
       ptrs=[]

       # Argument names (buffer or constant node)
       args=[]
       inargs=[]
       outargs=[]

       argsStr=""
       inArgsStr=""
       outArgsStr=""
       for io in self.inputNames:
            ioObj = self._inputs[io] 
            if ioObj.constantNode:
               # Argument is name of constant Node
               args.append(ioObj.constantNode.name)
               inargs.append(ioObj.constantNode.name)
            else:
               # Argument is a buffer created from FIFO
               buf = "i%d" % theId
               ptrs.append(buf)
               args.append(buf)
               inargs.append(buf)
               # Buffer and fifo
               inputs.append((buf,self.listOfargs[theId]))
            theId = theId + 1
       for io in self.outputNames:
            buf = "o%d" % theId
            ptrs.append(buf)
            args.append(buf)
            outargs.append(buf)
            outputs.append((buf,self.listOfargs[theId]))
            theId = theId + 1

       argsStr="".join(joinit(args,","))
       inArgsStr="".join(joinit(inargs,","))
       outArgsStr="".join(joinit(outargs,","))

       if ctemplate:
           result=Dsp.CTEMPLATE.render(func=self._nodeName,
            theType = theType,
            nb = theLen,
            ptrs = ptrs,
            args = argsStr,
            inputs=inputs, 
            outputs=outputs,
            node=self
            )
       else:
           result=Dsp.PYTEMPLATE.render(func=self._nodeName,
            theType = theType,
            nb = theLen,
            ptrs = ptrs,
            args = argsStr,
            inArgs= inArgsStr, 
            outArgs= outArgsStr, 
            inputs=inputs, 
            outputs=outputs,
            node=self
            )
       return(result)

class Unary(GenericFunction):
    def __init__(self,funcname,theType,length):
        GenericFunction.__init__(self,funcname,theType,length)

        self.addInput("i",theType,length)
        self.addOutput("o",theType,length)

class Binary(GenericFunction):
    def __init__(self,funcname,theType,length):
        GenericFunction.__init__(self,funcname,theType,length)

        self.addInput("ia",theType,length)
        self.addInput("ib",theType,length)
        
        self.addOutput("o",theType,length)


BINARYOP=["scale","add","and","mult","not","or","sub","xor","cmplx_mult_cmplx","cmplx_mult_real"
]

class Dsp(GenericFunction):

    def __init__(self,name,theType,length):
        # Some different graph functions correspond to the same
        # DSP function like IFFT
        # So we rename the cmsis function to call the same function
        
        cmsisname = "arm_%s_%s" % (name,theType.dspExtension)
        GenericFunction.__init__(self, cmsisname,theType,length)
        
        if name in BINARYOP:
            self.addInput("ia",theType,length)
            self.addInput("ib",theType,length)
        else:
           self.addInput("i",theType,length)
        
        self.addOutput("o",theType,length)
    

    @property
    def typeName(self):
        return "CMSIS-DSP"
  
