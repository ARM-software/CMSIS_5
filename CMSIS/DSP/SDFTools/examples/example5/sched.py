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


FIFOSIZE0=384
buf0=np.zeros(FIFOSIZE0,dtype=np.int16)

FIFOSIZE1=768
buf1=np.zeros(FIFOSIZE1,dtype=np.int16)

FIFOSIZE2=1024
buf2=np.zeros(FIFOSIZE2,dtype=np.int16)

FIFOSIZE3=377
buf3=np.zeros(FIFOSIZE3,dtype=np.int16)

FIFOSIZE4=754
buf4=np.zeros(FIFOSIZE4,dtype=np.int16)


def scheduler(mfccConfig,dispbuf):
    sdfError=0
    nbSchedule=0
    debugCounter=12

    #
    #  Create FIFOs objects
    #
    fifo0=FIFO(FIFOSIZE0,buf0)
    fifo1=FIFO(FIFOSIZE1,buf1)
    fifo2=FIFO(FIFOSIZE2,buf2)
    fifo3=FIFO(FIFOSIZE3,buf3)
    fifo4=FIFO(FIFOSIZE4,buf4)

    # 
    #  Create node objects
    #
    audioWin = SlidingBuffer(1024,256,fifo1,fifo2)
    mfcc = MFCC(1024,13,fifo2,fifo3,mfccConfig)
    mfccWin = SlidingBuffer(754,377,fifo3,fifo4)
    sink = NumpySink(754,fifo4,dispbuf)
    src = WavSource(384,fifo0,True,"test_stereo.wav")
    toMono = StereoToMono(384,192,fifo0,fifo1)

    while((sdfError==0) and (debugCounter > 0)):
       nbSchedule = nbSchedule + 1

       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = src.run()
       if sdfError < 0:
          break
       sdfError = toMono.run()
       if sdfError < 0:
          break
       sdfError = audioWin.run()
       if sdfError < 0:
          break
       sdfError = mfcc.run()
       if sdfError < 0:
          break
       sdfError = mfccWin.run()
       if sdfError < 0:
          break
       sdfError = sink.run()
       if sdfError < 0:
          break

       debugCounter = debugCounter - 1 
    return(nbSchedule,sdfError)
