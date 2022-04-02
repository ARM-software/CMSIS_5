#
# Generated with CMSIS-DSP SDF Scripts.
# The generated code is not covered by CMSIS-DSP license.
# 
# The support classes and code is covered by CMSIS-DSP license.
#

import sys


import numpy as np
import cmsisdsp as dsp
from cmsisdsp.sdf.nodes.simu import *
from appnodes import * 
from custom import *

DEBUGSCHED=False

# 
# FIFO buffers
# 


FIFOSIZE0=128
buf0=np.zeros(FIFOSIZE0,dtype=np.int16)

FIFOSIZE1=128
buf1=np.zeros(FIFOSIZE1,dtype=np.int16)


def scheduler():
    sdfError=0
    nbSchedule=0

    #
    #  Create FIFOs objects
    #
    fifo0=FIFO(FIFOSIZE0,buf0)
    fifo1=FIFO(FIFOSIZE1,buf1)

    # 
    #  Create node objects
    #
    proc = Processing(128,128,fifo0,fifo1)
    sink = VHTSink(128,fifo1,0)
    src = VHTSource(128,fifo0,0)

    while(sdfError==0):
       nbSchedule = nbSchedule + 1

       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = proc.run()
       if sdfError < 0:
          break
       sdfError = sink.run()
       if sdfError < 0:
          break

    return(nbSchedule,sdfError)
