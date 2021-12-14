###########################################
# Project:      CMSIS DSP Library
# Title:        description.py
# Description:  Schedule generation
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
"""Description of the graph"""

import networkx as nx
import numpy as np 
from sympy import Matrix
from sympy.core.numbers import ilcm,igcd

import sdf.schedule.graphviz
import sdf.schedule.ccode
import sdf.schedule.pythoncode

from sdf.schedule.node import *
from sdf.schedule.config import *
from sdf.schedule.types import *

class IncompatibleIO(Exception):
    pass

class GraphIsNotConnected(Exception):
    pass

class NotSchedulableError(Exception):
    pass

class DeadlockError(Exception):
    pass

class CannotDelayConstantError(Exception):
    pass

class FifoBuffer:
    """Buffer used by a FIFO"""
    def __init__(self,bufferID,theType,length):
        self._length=length 
        self._theType=theType 
        self._bufferID=bufferID

class FIFODesc:
    """A FIFO connecting two nodes"""
    def __init__(self,fifoid):
        # The FIFO is in fact just an array
        self.isArray=False 
        # Max distance between a write and a read to the FIFO
        # If it is 1, data written to FIFO
        # is immediately read so buffer can be reused
        self.distance=1
        # FIFO length
        self.length=0
        # FIFO type
        self.theType=None 
        # Buffer used by FIFO
        self.buffer=None 
        self._fifoID=fifoid 
        # Source IO
        self.src = None 
        # Dest IO
        self.dst = None 
        # FIFO delay
        self.delay=0
        # Can use a shared buffer ?
        self.isShared = False

        self._writeTime= 0

        # Track when FIFO is used
        # For allocation of shared buffers we need
        # to know when 2 buffers are used at same time
        self._timeOfUse=[]

        # shared buffer number not yet allocated
        self.sharedNB=-1

    # For c code generation 
    @property
    def isArrayAsInt(self):
        if self.isArray:
            return(1)
        else:
            return(0)

    @property
    def hasDelay(self):
        return(self.delay>0)

    def dump(self):
        shared=0 
        if self.isShared:
            shared=1
        print("array %d dist %d len %d %s id %d src %s:%s dst %s:%s  shared:%d" % 
            (self.isArray,
             self.distance,
             self.length,
             self.theType.ctype,
             self.fifoID,
             self.src.owner.nodeID,
             self.src.name,
             self.dst.owner.nodeID,
             self.dst.name,
             shared))

    @property
    def fifoID(self):
        return self._fifoID
    
    def recordWrite(self,theTime):
        self._timeOfUse.append(theTime)
        if self._writeTime == 0:
           self._writeTime = theTime

    def recordRead(self,theTime):
        self._timeOfUse.append(theTime)
        delta = theTime - self._writeTime 
        self._writeTime = 0
        if delta > self.distance:
            self.distance = delta

def analyzeStep(vec,allFIFOs,theTime):
    """Analyze an evolution step to know which FIFOs are read and written to"""
    fifoID = 0 
    for fifo in (vec > 0):
        if fifo:
            allFIFOs[fifoID].recordWrite(theTime) 
        fifoID = fifoID + 1

    fifoID = 0 
    for fifo in (vec < 0):
        if fifo:
            allFIFOs[fifoID].recordRead(theTime) 
        fifoID = fifoID + 1

class Graph():

    def __init__(self):
        self._nodes={}
        self._edges={}
        self._delays={}
        self._constantEdges={}
        self._g = nx.Graph()
        self._sortedNodes=None
        self._totalMemory=0
        self._allFIFOs = None 
        self._allBuffers = None

    def connect(self,nodea,nodeb):
        # When connecting to a constant node we do nothing
        # since there is no FIFO in this case
        # and it does not participate to the scheduling.
        if (isinstance(nodea,Constant)):
            nodeb.constantNode = nodea
            self._constantEdges[(nodea,nodeb)]=True
        else:
            if nodea.compatible(nodeb):
                self._sortedNodes = None
                self._sortedEdges = None
                self._g.add_edge(nodea.owner,nodeb.owner)
    
                nodea.fifo = (nodea,nodeb) 
                nodeb.fifo = (nodea,nodeb)
                self._edges[(nodea,nodeb)]=True
                if not (nodea.owner in self._nodes):
                   self._nodes[nodea.owner]=True
                if not (nodeb.owner in self._nodes):
                   self._nodes[nodeb.owner]=True
            else:
                raise IncompatibleIO

    def connectWithDelay(self,nodea,nodeb,delay):
        # We cannot connect with delay to a constant node
        if (isinstance(nodea,Constant)):
            raise CannotDelayConstantError
        else:
            self.connect(nodea,nodeb)
            self._delays[(nodea,nodeb)] = delay
    
    def __str__(self):
        res=""
        for (a,b) in self._edges: 
            nodea = a.owner
            nodeb = b.owner 

            res += ("%s.%s -> %s.%s\n" % (nodea.nodeID,a.name, nodeb.nodeID,b.name))

        return(res)

    def initializeFIFODescriptions(self,config,allFIFOs, fifoLengths):
        """Initialize FIFOs datastructure""" 
        for fifo in allFIFOs:
            edge = self._sortedEdges[fifo.fifoID]
            fifo.length = fifoLengths[fifo.fifoID]
            src,dst = edge
            fifo.src=src
            fifo.dst=dst 
            fifo.delay=self.getDelay(edge)
            if src.nbSamples == dst.nbSamples:
                if fifo.delay==0:
                   fifo.isArray = True 
                if fifo.distance==1:
                    if fifo.delay==0:
                       fifo.isShared = True
            fifo.theType = src.theType
            #fifo.dump()

        # When we have bufA -> Node -> bufB then
        # bufA and bufB can't share the same memory.

        # For the allocation of shared buffer we scan all times 
        # of use and for each time we look at the FIFOs which 
        # can be potentially shared.

        # For each time of use, record the potentially shareable fifos
        # which are used
        fifoForTime={} 
        for fifo in allFIFOs:
            if fifo.isShared:
                for t in fifo._timeOfUse:
                    if t in fifoForTime:
                        fifoForTime[t].append(fifo)
                    else:
                        fifoForTime[t]=[fifo]

        # If several shareable FIFOs are used at same time, they
        # must be assigned to different shared buffers if possible
        usedAtSameTime={} 
        for t in sorted(fifoForTime.keys()):
            if len(fifoForTime[t])>2:
                # This case is not managed in this version.
                # It could occur with quadripoles for instance
                for fifo in fifoForTime[t]:
                    fifo.isShared=False
            elif len(fifoForTime[t])==2:
                 # 2 FIFOs are used at the same time
                 fifoA = fifoForTime[t][0]
                 fifoB = fifoForTime[t][1]
                 if fifoA.sharedNB >= 0 and fifoA.sharedNB == fifoB.sharedNB:
                     # Those FIFOs are already both assigned to a buffer
                     # and we can't reassign and they use the same buffer
                     # So we can't consistently associate shared buffers to those
                     # FIFOs
                     fifoA.isShared=False 
                     fifoB.isShared=False
                 else:
                     # The 2 FIFOs were never associated with a shared buffer
                     if fifoA.sharedNB < 0 and fifoB.sharedNB < 0:
                         fifoA.sharedNB=0 
                         fifoB.sharedNB=1 
                     # One FIFO is associated, so we associate the other one
                     # to the other shared buffer
                     elif fifoA.sharedNB < 0:
                         fifoA.sharedNB = 1 - fifoB.sharedNB 
                     else:
                         fifoB.sharedNB = 1 - fifoA.sharedNB 
            else:
                fifoA = fifoForTime[t][0]
                if fifoA.sharedNB < 0:
                    fifoA.sharedNB = 0
                        


        # Now we create buffers 
        maxSharedA=0
        maxSharedB=0
        allBuffers=[]
        for fifo in allFIFOs:
            lengthInBytes = fifo.theType.bytes * fifo.length
            if fifo.isShared:
                if fifo.sharedNB == 0:
                    if lengthInBytes > maxSharedA:
                       maxSharedA = lengthInBytes 
                if fifo.sharedNB == 1:
                    if lengthInBytes > maxSharedB:
                       maxSharedB = lengthInBytes 

        bufferID=0
        sharedA=None
        sharedB=None
        if maxSharedA > 0 and config.memoryOptimization:
           # Create the shared buffer if memory optimization on
           sharedA = FifoBuffer(bufferID,CType(UINT8),maxSharedA)
           allBuffers.append(sharedA)
           bufferID = bufferID + 1

        if maxSharedB > 0 and config.memoryOptimization:
           # Create the shared buffer if memory optimization on
           sharedB = FifoBuffer(bufferID,CType(UINT8),maxSharedB)
           allBuffers.append(sharedB)
           bufferID = bufferID + 1

        for fifo in allFIFOs:
            # Use shared buffer if memory optimization
            if fifo.isShared and config.memoryOptimization:
                if fifo.sharedNB == 0:
                   fifo.buffer=sharedA 
                else:
                    fifo.buffer=sharedB
            else:
                buf = FifoBuffer(bufferID,fifo.theType,fifo.length)
                allBuffers.append(buf)
                fifo.buffer=buf
                bufferID = bufferID + 1

        self._totalMemory = 0
        for buf in allBuffers:
            self._totalMemory = self._totalMemory + buf._theType.bytes * buf._length

        #for fifo in allFIFOs:
        #    fifo.dump()
        return(allBuffers)




    @property
    def constantEdges(self):
        return list(self._constantEdges.keys())
    
    @property
    def nodes(self):
        return list(self._nodes.keys())

    @property
    def edges(self):
        return list(self._edges.keys())
    

    def hasDelay(self,edge):
        return(edge in self._delays)

    def getDelay(self,edge):
        if self.hasDelay(edge):
           return(self._delays[edge])
        else:
           return(0)

    def checkGraph(self):
        if not nx.is_connected(self._g):
            raise GraphIsNotConnected

    def topologyMatrix(self):
        self.checkGraph()
        rows=[]
        # This is used in schedule generation
        # and all functions must use the same node ordering
        self._sortedNodes = sorted(self.nodes, key=lambda x: x.nodeID)
        # Arbitrary order but used for now
        self._sortedEdges = self.edges.copy()
        #for x in self._sorted:
        #    print(x.nodeID)

        for edge in self._sortedEdges: 
            na,nb = edge
            currentRow=[0] * len(self._sortedNodes) 

            ia=self._sortedNodes.index(na.owner)
            ib=self._sortedNodes.index(nb.owner)

            # Produced by na on the edge
            currentRow[ia] = na.nbSamples

            # Consumed by nb on the edge
            currentRow[ib] = -nb.nbSamples

            rows.append(currentRow)

        return(np.array(rows))

    def nullVector(self):
        m = self.topologyMatrix()
        r=Matrix(m).nullspace()
        if len(r) != 1:
           raise NotSchedulableError
        result=list(r[0])
        denominators = [x.q for x in result]
        # Remove denominators
        ppcm = ilcm(*denominators)
        #print(ppcm)
        intValues = [x * ppcm for x in result]
        # Convert intValues to the smallest possible values
        gcd = igcd(*intValues)
        return([x / gcd for x in intValues])

    @property
    def initEvolutionVector(self):
        """Initial FIFO state taking into account delays"""
        return(np.array([self.getDelay(x) for x in self.edges]))

    def evolutionVectorForNode(self,nodeID):
        """Return the evolution vector corresponding to a selected node"""
        v = np.zeros(len(self._sortedNodes))
        v[nodeID] = 1 
        return(v)

    def computeSchedule(self,config=Configuration()):
        # Init values
        initB = self.initEvolutionVector
        initN = self.nullVector()

        # Current values (copys)
        b = np.array(initB)
        n = np.array(initN)

        if config.displayFIFOSizes:
           for edge in self._sortedEdges:
             print("%s:%s -> %s:%s" % (edge[0].owner.nodeID,edge[0].name,edge[1].owner.nodeID,edge[1].name))
           print(b)

        # Topology matrix
        t = np.array(self.topologyMatrix())

        # Define the list of FIFOs objects
        nbFIFOS = t.shape[0]
        allFIFOs = [] 
        for i in range(nbFIFOS):
            allFIFOs.append(FIFODesc(i))

        # Normalization vector
        normV =  1.0*np.apply_along_axis(abs,1,t).max(axis=1)

        # bMax below is used to track maximum FIFO size 
        # occuring during a run of the schedule
        #
        # The heuristric is:
        #
        # First we compute on each edge the maximum absolute value
        # It is the minimum amount of data an edge must contain
        # for the system to work either because it is produced
        # by a node or consumed by another.
        # We use this value as an unit of measurement for the edge.
        # So, we normalize the FIFO lengths by this size.
        # If this occupancy number is > 1 then it means
        # that enough data is available on the FIFO for the
        # consumer to consume it.
        # When we select a node for scheduling later we try
        # to minimize the occupancy number of all FIFOs by
        # selecting the schedulign which is giving the
        # minimum maximum occupancy number after the run.
        bMax = 1.0*np.array(initB) / normV


        schedule=[]

        zeroVec = np.zeros(len(self._sortedNodes))
        evolutionTime = 0
        # While there are remaining nodes to schedule
        while (n != zeroVec).any():
            # Look for the best mode to schedule
            # which is the one giving the minimum FIFO increase
            
            # None selected
            selected = -1

            # Min FIFO size found
            minVal = 10000000
            nodeID = 0
            for node in self._sortedNodes:
                # If the node can be scheduled
                if n[nodeID] > 0:
                   # Evolution vector if this node is selected
                   v = self.evolutionVectorForNode(nodeID)
                   # New fifos size after this evolution
                   newB = np.dot(t,v) + b

                   # Check that there is no FIFO underflow:
                   if np.all(newB >= 0):
                      # Total FIFO size for this possible execution
                      # We normalize to get the occupancy number as explained above
                      theMin = (1.0*np.array(newB) / normV).max()
                      # If this possible evolution is giving smaller FIFO size
                      # (measured in occupancy number) then it is selected
                      if theMin <= minVal:
                         minVal = theMin
                         selected = nodeID 

                nodeID = nodeID + 1

            # No node could be scheduled because of not enough data
            # in the FIFOs. It should not occur if there is a null
            # space of dimension 1. So, it is probably a bug if
            # this exception is raised
            if selected < 0:
               raise DeadlockError
            # Now  we have evaluated all schedulable nodes for this run
            # and selected the one giving the smallest FIFO increase

            # Real evolution vector for selected node
            evol =  self.evolutionVectorForNode(selected)
            # Keep track that this node has been schedule
            n = n - evol
            # Compute new fifo state
            fifoChange = np.dot(t,evol)
            b = fifoChange + b

            if config.displayFIFOSizes:
               print(b)
            
            bMax = np.maximum(b,bMax)
            schedule.append(selected)

            # Analyze FIFOs to know if a FIFOs write is
            # followed immediately by a FIFO read of same size
            analyzeStep(fifoChange,allFIFOs,evolutionTime)
            evolutionTime = evolutionTime + 1

        fifoMax=np.floor(bMax).astype(np.int32)
        
        allBuffers=self.initializeFIFODescriptions(config,allFIFOs,fifoMax)
        self._allFIFOs = allFIFOs 
        self._allBuffers = allBuffers
        return(Schedule(self,self._sortedNodes,self._sortedEdges,schedule))


class Schedule:
    def __init__(self,g,sortedNodes,sortedEdges,schedule):
        self._sortedNodes=sortedNodes
        self._sortedEdges=sortedEdges
        self._schedule = schedule 
        self._graph = g
        # Nodes containing pure functions (no state) like some
        # CMSIS-DSP functions.
        # When scheduling is using the option codeArray, the
        # schedule is encoded as an array.
        # Function calls cannot be inlined anymore and we need
        # to create new nodes for those function calls.
        # The pureNode structure is done for this.
        # It is a record because we want to reuse nodes for same
        # types.
        self._pureNodes={}
        nodeCodeID = 0
        pureClassID = 1
        for n in self.nodes:
            n.codeID = nodeCodeID
            nodeCodeID = nodeCodeID + 1
            # Constant nodes are ignored since they have
            # no arcs, and are connected to no FIFOs
            theArgs=[] 
            theArgTypes=[]
            i,o=n.allIOs()
            for io in i:
                # An io connected to a constant node has no fifo 
                if not io.fifo is None:
                   theArgs.append(self.fifoID(io.fifo))
                   theArgTypes.append(io.ctype)
                else:
                # Instead the arg is the name of a constant node
                # instead of being a fifo ID
                   theArgs.append(io.constantNode.name)
                   theArgTypes.append(io.constantNode.name)
            for io in o:
                theArgs.append(self.fifoID(io.fifo))
                theArgTypes.append(io.ctype)
            n.args=theArgs

            # Analyze the nature of arguments for pure functions
            # The information created during this analysis
            # is useful when generating a class containing the
            # pure function
            if not n.hasState:
               theType=(n.nodeName,tuple(theArgTypes))
               if not theType in self._pureNodes:
                  self._pureNodes[theType]=n
                  n.pureClassID = pureClassID 
                  pureClassID = pureClassID + 1
               else:
                  n.pureClassID = self._pureNodes[theType].pureClassID
               n.pureNodeType=theType
               n.analyzeArgs()

    def hasDelay(self,edge):
        return(self._graph.hasDelay(edge))

    def getDelay(self,edge):
        return(self._graph.getDelay(edge))

    @property
    def pureNodes(self):
        return self._pureNodes
    

    @property
    def constantEdges(self):
        return self._graph.constantEdges

    @property
    def nodes(self):
        return self._sortedNodes

    @property
    def edges(self):
        return self._sortedEdges

    @property
    def schedule(self):
        return self._schedule

    #@property
    #def fifoLengths(self):
    #    return self._fifos

    @property 
    def scheduleLength(self):
        return len(self.schedule)

    @property 
    def memory(self):
        #theBytes=[x[0].theType.bytes for x in self.edges]
        #theSizes=[x[0]*x[1] for x in zip(self.fifoLengths,theBytes)]
        #return(np.sum(theSizes))
        return(self._graph._totalMemory)

    @property
    def graph(self):
        return self._graph

    def fifoID(self,edge):
        return(self.edges.index(edge))

    def outputFIFOs(self,node):
        outs=[]
        for io in node.outputNames:
            x = node._outputs[io]
            fifo=(self.fifoID(x.fifo),io)
            outs.append(fifo)
            
        return(outs)

    def ccode(self,directory,config=Configuration()):
        """Write graphviz into file f""" 
        sdf.schedule.ccode.gencode(self,directory,config)

    def pythoncode(self,directory,config=Configuration()):
        """Write graphviz into file f""" 
        sdf.schedule.pythoncode.gencode(self,directory,config)

    def graphviz(self,f,config=Configuration()):
        """Write graphviz into file f""" 
        sdf.schedule.graphviz.gengraph(self,f,config)
    
    
    
    