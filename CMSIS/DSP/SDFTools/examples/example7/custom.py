from cmsisdsp.sdf.nodes.simu import *

import numpy as np 
import cmsisdsp as dsp


class Processing(GenericNode):
    def __init__(self,inputSize,outputSize,fifoin,fifoout):
        GenericNode.__init__(self,inputSize,outputSize,fifoin,fifoout)

    def run(self):

        i=self.getReadBuffer()
        o=self.getWriteBuffer()

        b=dsp.arm_scale_q15(i,0x6000,1)

        o[:]=b[:]

        
        return(0)