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


FIFOSIZE0=160
buf0=np.zeros(FIFOSIZE0,dtype=np.int16)

FIFOSIZE1=400
buf1=np.zeros(FIFOSIZE1,dtype=np.int16)

FIFOSIZE2=49
buf2=np.zeros(FIFOSIZE2,dtype=np.int16)

FIFOSIZE3=98
buf3=np.zeros(FIFOSIZE3,dtype=np.int16)

FIFOSIZE4=98
buf4=np.zeros(FIFOSIZE4,dtype=np.int16)

FIFOSIZE5=1
buf5=np.zeros(FIFOSIZE5,dtype=np.int16)


def scheduler(input_array,window,coef_q15,coef_shift,intercept_q15,intercept_shift):
    sdfError=0
    nbSchedule=0
    debugCounter=13

    #
    #  Create FIFOs objects
    #
    fifo0=FIFO(FIFOSIZE0,buf0)
    fifo1=FIFO(FIFOSIZE1,buf1)
    fifo2=FIFO(FIFOSIZE2,buf2)
    fifo3=FIFO(FIFOSIZE3,buf3)
    fifo4=FIFO(FIFOSIZE4,buf4)
    fifo5=FIFO(FIFOSIZE5,buf5)

    # 
    #  Create node objects
    #
    audioWin = SlidingBuffer(400,240,fifo0,fifo1)
    feature = Feature(400,1,fifo1,fifo2,window)
    featureWin = SlidingBuffer(98,49,fifo2,fifo3)
    fir = FIR(98,98,fifo3,fifo4)
    kws = KWS(98,1,fifo4,fifo5,coef_q15,coef_shift,intercept_q15,intercept_shift)
    sink = Sink(1,fifo5)
    src = Source(160,fifo0,input_array)

    while((sdfError==0) and (debugCounter > 0)):
       nbSchedule = nbSchedule + 1

       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = feature.run()
       if sdfError < 0:
          break
       sdfError = featureWin.run()
       if sdfError < 0:
          break
       sdfError = fir.run()
       if sdfError < 0:
          break
       sdfError = kws.run()
       if sdfError < 0:
          break
       sdfError = sink.run()
       if sdfError < 0:
          break

       debugCounter = debugCounter - 1 
    return(nbSchedule,sdfError)
