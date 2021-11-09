/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        RingPrivate.h
 * Description:  Implementation for RTX + Keil MDK
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
#ifndef _RINGPRIVATE_H_
#define _RINGPRIVATE_H_

/*

Implementation for RTX + Keil MDK Event logger

*/

#include <stddef.h>
#include "arm_vsi.h"
#ifdef _RTE_
#include "RTE_Components.h"
#endif
#include CMSIS_device_header

#include "cmsis_os2.h"

#include "SchedEvents.h"
/*

RTX dependent definition

*/
#define RING_BEGINCRITICALSECTION()  NVIC_DisableIRQ ((IRQn_Type)config->interruptID); 
#define RING_ENDCRITICALSECTION() NVIC_EnableIRQ ((IRQn_Type)config->interruptID);

#define RING_WAIT_BUFFER(TIMEOUT) osThreadFlagsWait(1,osFlagsWaitAny,(TIMEOUT))
#define RING_HASWAITERROR(F) (F < 0)

#define RING_RELEASE_BUFFER(THREADID) osThreadFlagsSet((osThreadId_t)(THREADID),1)

/* Debug trace using Event Recorder */
#define RING_DBG_USER_RESERVE_BUFFER(ID,CONF) EventRecord2 (Evt_UsrReserve, (ID), (uint32_t)(CONF))
#define RING_DBG_USER_RELEASE_BUFFER(ID,CONF) EventRecord2 (Evt_UsrRelease, (ID), (uint32_t)(CONF))
#define RING_DBG_USER_WAIT_BUFFER(ID,CONF) EventRecord2 (Evt_UsrWait, (ID), (uint32_t)(CONF))
#define RING_DBG_USER_BUFFER_RELEASED(ID,CONF) EventRecord2 (Evt_UsrFree, (ID), (uint32_t)(CONF))
#define RING_DBG_USER_STATUS(SA,SB,CONF) EventRecord4 (Evt_UsrStatus, config->SA,config->SB,(uint32_t)(CONF),0)

#define RING_DBG_INT_RESERVE_BUFFER(ID,CONF) EventRecord2 (Evt_IntReserve, (ID), (uint32_t)(CONF))
#define RING_DBG_INT_RELEASE_BUFFER(ID,CONF) EventRecord2 (Evt_IntRelease, (ID), (uint32_t)(CONF))
#define RING_DBG_INT_RELEASE_USER(CONF) EventRecord2 (Evt_IntReleaseUser, (uint32_t)(CONF), 0)
#define RING_DBG_INT_STATUS(SA,SB,CONF) EventRecord4 (Evt_IntStatus, config->SA,config->SB,(uint32_t)(CONF),0)

#define RING_DBG_ERROR(ERROR,CONF) EventRecord2 (Evt_Error, (ERROR), (uint32_t)(CONF))

#endif