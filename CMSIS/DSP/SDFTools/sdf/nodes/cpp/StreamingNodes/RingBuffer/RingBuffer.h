/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        RingBuffer.h
 * Description:  Ring buffer to connect the SDF to audio sources and sinks
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
#ifndef _RINGBUFFER_H_
#define _RINGBUFFER_H_


#ifdef   __cplusplus
extern "C"
{
#endif

typedef enum {
 kErrorOverflowUnderflow=-1,
 kTimeout=-2,
 kErrorHWOverlowUnderflow=-3,
 kNoError=0
} ring_error_t;

typedef struct {
  uint8_t *buffer;
  uint32_t userBufferStatus;
  uint32_t intBufferStatus;

  int32_t nbBuffers;
  uint32_t bufferSize;

  int32_t interruptBufferIDStart;
  int32_t interruptBufferIDStop;
  int32_t userBufferIDStart;
  int32_t userBufferIDStop;
  int32_t waiting;
  int timeout;
  ring_error_t error;
  int interruptID;
} ring_config_t;


  /**
   * @brief  Ring buffer initialization
   * @param[in, out] buf         ring buffer configuration.
   * @param[in]      nbBuffers   number of buffers (max 32)
   * @param[in]      bufferSize  size of each buffer in bytes
   * @param[in]      buffer      array for the buffer storage (bufferSize*nbBuffers)
   * @param[in]      interruptID interrupt ID
   * @param[in]      timeout     timeout (meaning is RTOS dependent)
   * @return  Nothing
   */
  void ringInit(ring_config_t *buf,
    uint32_t nbBuffers,
    uint32_t bufferSize,
    uint8_t *buffer,
    int interruptID,
    int timeout);



void ringClean(ring_config_t *buf);

/*

Try to reserve a buffer from a user thread.

*/
int ringUserReserveBuffer(ring_config_t *buf);

/*

Release a buffer from user htread

*/
void ringUserReleaseBuffer(ring_config_t *buf);

/*

Reserve a buffer from interrupt

*/
int ringInterruptReserveBuffer(ring_config_t *buf);

/*

Release a buffer from interrupt 

*/
void ringInterruptReleaseBuffer(ring_config_t *buf,void *threadId);

/*

Get address of buffer

*/

uint8_t *ringGetBufferAddress(ring_config_t *buf,int id);


#ifdef   __cplusplus
}
#endif

#endif /* #ifndef _RINGBUFFER_H_*/