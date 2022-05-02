from ..simu import *
import numpy as np 
import cmsisdsp as dsp
from .VHT import *


class VHTSource(GenericSource):
    def __init__(self,outputSize,fifoout,theid):
        GenericSource.__init__(self,outputSize,fifoout)
        self._src_=Source(theid,outputSize)
        self._src_.wait()

    def run(self):
        o=self.getWriteBuffer()
        
        theInput = self._src_.get()
        o[:]=theInput[:]

        
        return(0)

    def __del__(self):
        self._src_.end()

class VHTSink(GenericSink):
    def __init__(self,inputSize,fifoin,theid):
        GenericSink.__init__(self,inputSize,fifoin)
        self._sink_=Sink(theid,inputSize)
        self._sink_.wait()

    def run(self):
        i=self.getReadBuffer()
        self._sink_.put(i)

        return(0)

    def __del__(self):
        self._sink_.end()