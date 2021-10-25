/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        RingInit.cpp
 * Description:  Initialization of the ring data structure for an audio source
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
#include "arm_math.h"
#include "RingConfig.h"
#include "RingInit.h"
#include "RingBuffer.h"




int initRingAndAudio(ring_config_t *ringConfigRX,
    uint8_t *rxBuffer,
    int rxInterruptID,
    ring_config_t *ringConfigTX, 
    uint8_t *txBuffer,
    int txInterruptID,
    int timeOut)
{
  /* Initialization of the ring buffer data structure */
  if (ringConfigRX != NULL)
  {
     ringInit(ringConfigRX,RING_NBBUFS,RING_BUFSIZE,rxBuffer,rxInterruptID,timeOut);
  }

  if (ringConfigTX != NULL)
  {
     ringInit(ringConfigTX,RING_NBBUFS,RING_BUFSIZE,txBuffer,txInterruptID,timeOut);
  }

  /* Initialization of the audio HW and reservation of first buffer from the
  ring buffer
  */
  int err=AudioDrv_Setup();
  return(err);
}