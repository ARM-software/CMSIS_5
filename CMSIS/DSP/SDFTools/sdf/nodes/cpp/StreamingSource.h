/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        StreamingSource.h
 * Description:  Streaming source working with the Ring buffer
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
#ifndef _STREAMING_SOURCE_H_ 
#define _STREAMING_SOURCE_H_ 

#include "RingBuffer.h"


template<typename OUT,int outputSize>
class StreamingSource: public GenericSource<OUT,outputSize>
{
public:
    StreamingSource(FIFOBase<OUT> &dst,ring_config_t *config):
    GenericSource<OUT,outputSize>(dst),mConfig(config){};

    int run(){
        OUT *b=this->getWriteBuffer();
        /* 
           Try to reserve a buffer. If no buffer is available, the task running
           this node will sleep.

           If there is a timeout (configured when the ring buffer was initialized)
           the function will return a NULL pointer.

        */
        int bufID=ringUserReserveBuffer(mConfig);
        uint8_t *buf=ringGetBufferAddress(mConfig,bufID);


        if (buf != NULL)
        {
           /* If a buffer is available we copy the data to the FIFO
           */
           memcpy((void*)b,buf,outputSize*sizeof(OUT));

           /* We release the buffer so than it can be used by the interrupt */
           ringUserReleaseBuffer(mConfig);
           return(0);
        }
        else 
        {
            return(mConfig->error);
        }
    };

protected:
    ring_config_t *mConfig;


};
#endif /* _STREAMING_SOURCE_H_ */