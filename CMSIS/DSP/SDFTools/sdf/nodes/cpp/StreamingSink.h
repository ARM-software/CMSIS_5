/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        StreamingSink.h
 * Description:  Streaming Sink working with the RingBuffer
 *
 * $Date:        30 July 2021
 * $Revision:    V1.10.0
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _STREAMING_SINK_H_ 
#define _STREAMING_SINK_H_ 

#include "RingBuffer.h"


template<typename IN, int inputSize>
class StreamingSink: public GenericSink<IN, inputSize>
{
public:
    StreamingSink(FIFOBase<IN> &src,ring_config_t *config):
    GenericSink<IN,inputSize>(src),mConfig(config){};

    int run()
    {
        IN *b=this->getReadBuffer();

        int bufID=ringUserReserveBuffer(mConfig);
        uint8_t *buf=ringGetBufferAddress(mConfig,bufID);

        if (buf != NULL)
        {
           /* If a buffer is available we copy the data to the FIFO
           */
           memcpy(buf,(void*)b,inputSize*sizeof(IN));

           /* We release the buffer so than it can be used by the interrupt */
           ringUserReleaseBuffer(mConfig);
           return(0);
        }
        else 
        {
            return(mConfig->error);
        }

        return(0);
    }
protected:
    ring_config_t *mConfig;
};

#endif /* _STREAMING_SINK_H_ */