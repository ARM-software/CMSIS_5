/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        RingBuffer.cpp
 * Description:  Implementation of the Ring buffer.
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

/* 

RTOS dependent definitions must be in RingPrivate.h.
Without a RingPrivate.h, this code cannot work.

 */
#include "RingPrivate.h"
#include "RingBuffer.h"

/*

RTOS Integration

*/
#ifndef RING_BEGINCRITICALSECTION
#define RING_BEGINCRITICALSECTION() 
#endif 

#ifndef RING_ENDCRITICALSECTION
#define RING_ENDCRITICALSECTION() 
#endif

#ifndef RING_WAIT_BUFFER
#define RING_WAIT_BUFFER(ID) 0
#endif

#ifndef RING_RELEASE_BUFFER
#define RING_RELEASE_BUFFER(THREADID) 
#endif

#ifndef RING_HASWAITERROR
#define RING_HASWAITERROR(ERR) 0 
#endif

/*

Debug integration

*/

#ifndef RING_DBG_USER_RESERVE_BUFFER
#define RING_DBG_USER_RESERVE_BUFFER(ID,CONF)
#endif 

#ifndef RING_DBG_USER_RELEASE_BUFFER
#define RING_DBG_USER_RELEASE_BUFFER(ID,CONF)
#endif

#ifndef RING_DBG_USER_WAIT_BUFFER
#define RING_DBG_USER_WAIT_BUFFER(ID,CONF)
#endif

#ifndef RING_DBG_USER_BUFFER_RELEASED
#define RING_DBG_USER_BUFFER_RELEASED(ID,CONF)
#endif

#ifndef RING_DBG_USER_STATUS
#define RING_DBG_USER_STATUS(SA,SB,CONF)
#endif

#ifndef RING_DBG_INT_RESERVE_BUFFER
#define RING_DBG_INT_RESERVE_BUFFER(ID,CONF)
#endif

#ifndef RING_DBG_INT_RELEASE_BUFFER
#define RING_DBG_INT_RELEASE_BUFFER(ID,CONF)
#endif

#ifndef RING_DBG_INT_RELEASE_USER
#define RING_DBG_INT_RELEASE_USER(CONF)
#endif

#ifndef RING_DBG_INT_STATUS
#define RING_DBG_INT_STATUS(SA,SB,CONF) 
#endif

#ifndef RING_DBG_ERROR
#define RING_DBG_ERROR(ERROR,CONF)
#endif

/*

Implementation

*/

#define RING_SET(FIELD,BIT) (config->FIELD) |= (1 << (config->BIT))
#define RING_CLEAR(FIELD,BIT) (config->FIELD) &= ~(1 << (config->BIT))
#define RING_TEST(FIELD,BIT) (((config->FIELD) & (1 << (config->BIT))) != 0)

#define RING_INC(ID)                  \
  config->ID++;                       \
  if (config->ID == config->nbBuffers)\
  {                                   \
     config->ID=0;                    \
  }

#define RING_BUSY(ID) \
  (RING_TEST(userBufferStatus,ID) || RING_TEST(intBufferStatus,ID))


void ringInit(ring_config_t *config,
    uint32_t nbBuffers,
    uint32_t bufferSize,
    uint8_t *buffer,
    int interruptID,
    int timeout)
{
  
  config->buffer=buffer;
  config->nbBuffers = nbBuffers;
  config->bufferSize = bufferSize;
  config->interruptBufferIDStart = 0;
  config->interruptBufferIDStop = 0;
  config->userBufferIDStart = 0;
  config->userBufferIDStop = 0;
  config->error=kNoError;
  config->waiting=0;
  config->timeout=timeout;

  config->interruptID = interruptID;
  config->userBufferStatus = 0;
  config->intBufferStatus = 0;
  
}

void ringClean(ring_config_t *config)
{
}

uint8_t *ringGetBufferAddress(ring_config_t *config,int id)
{
  if (id < 0)
  {
     return(NULL);
  }
  else
  {
      return(&config->buffer[id*config->bufferSize]);
  }
}

int ringInterruptReserveBuffer(ring_config_t *config)
{
   RING_DBG_INT_STATUS(userBufferStatus,intBufferStatus,config);
   if (config->error)
   {
     return(-1);
   }

   /* Try to reserve a buffer */
   if (RING_BUSY(interruptBufferIDStop))
   {
       /* If buffer is already used then kErrorOverflowUnderflow*/
       config->error=kErrorOverflowUnderflow;
       RING_DBG_ERROR(config->error,config);
       return(-1);
   }
   else 
   {
        RING_DBG_INT_RESERVE_BUFFER(config->interruptBufferIDStop,config);
        RING_SET(intBufferStatus,interruptBufferIDStop);
        RING_DBG_INT_STATUS(userBufferStatus,intBufferStatus,config);
        int id=config->interruptBufferIDStop;
        RING_INC(interruptBufferIDStop);
        return(id);
   }
}

void ringInterruptReleaseBuffer(ring_config_t *config,void *threadId)
{
     RING_DBG_INT_STATUS(userBufferStatus,intBufferStatus,config);
     if (config->error)
     {
        return;
     }
     if (config->interruptBufferIDStart != config->interruptBufferIDStop)
     {
         RING_DBG_INT_RELEASE_BUFFER(config->interruptBufferIDStart,config);
         RING_CLEAR(intBufferStatus,interruptBufferIDStart);
         /* Send release message in case the thread may be waiting */
         if (config->interruptBufferIDStart == config->userBufferIDStop)
         {
            if (config->waiting)
            {
                RING_DBG_INT_RELEASE_USER(config);
                RING_RELEASE_BUFFER(threadId);
            }
         }
         RING_INC(interruptBufferIDStart);
     }
} 

int ringUserReserveBuffer(ring_config_t *config)
{
   RING_BEGINCRITICALSECTION();
   RING_DBG_USER_STATUS(userBufferStatus,intBufferStatus,config);
   if (config->error)
   {
        RING_ENDCRITICALSECTION();
        return(-1);
   }
   /* If buffer is busy we wait*/
   if (RING_BUSY(userBufferIDStop))
   {
            config->waiting=1;
            RING_DBG_USER_WAIT_BUFFER(config->userBufferIDStop,config);
            RING_ENDCRITICALSECTION();

            int err = RING_WAIT_BUFFER(config->timeout);

            RING_BEGINCRITICALSECTION();
            RING_DBG_USER_BUFFER_RELEASED(config->userBufferIDStop,config);
            if (RING_HASWAITERROR(err))
            {
                RING_DBG_ERROR(err,config);
                config->error=kTimeout; 
                return(-1);
            }
    
   }
   
   RING_DBG_USER_RESERVE_BUFFER(config->userBufferIDStop,config);
   RING_SET(userBufferStatus,userBufferIDStop);
   int id=config->userBufferIDStop;
   RING_INC(userBufferIDStop);

   RING_ENDCRITICALSECTION();
   
   return(id);
}

void ringUserReleaseBuffer(ring_config_t *config)
{
     RING_BEGINCRITICALSECTION();
     RING_DBG_USER_STATUS(userBufferStatus,intBufferStatus,config);
     if (config->error)
     {
        RING_ENDCRITICALSECTION();
        return;
     }
     if (config->userBufferIDStart != config->userBufferIDStop)
     {
         RING_DBG_USER_RELEASE_BUFFER(config->userBufferIDStart,config);
         RING_CLEAR(userBufferStatus,userBufferIDStart);
         RING_INC(userBufferIDStart);
     }
     
     RING_ENDCRITICALSECTION();
}
