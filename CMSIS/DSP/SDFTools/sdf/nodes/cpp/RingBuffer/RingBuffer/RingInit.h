/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        RingInit.h
 * Description:  API to initialize the ring buffer
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
#ifndef _RINGINIT_H 
#define _RINGINIT_H



#include "RingBuffer.h"

#ifdef   __cplusplus
extern "C"
{
#endif

extern int32_t AudioDrv_Setup(void);
extern ring_config_t ringConfigRX;
extern ring_config_t ringConfigTX;
extern uint8_t* AudioRXBuffer();
extern uint8_t* AudioTXBuffer();


int initRingAndAudio(ring_config_t *ringConfigRX,
    uint8_t *rxBuffer,
    int rxInterruptID,
    ring_config_t *ringConfigTX, 
    uint8_t *txBuffer,
    int txInterruptID,
    int timeOut);

#ifdef   __cplusplus
}


#endif
#endif 
