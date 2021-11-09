/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        SchedEvents.h
 * Description:  Definition of the events for the Keil MDK Event logger
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
#ifndef _SCHEDEVT_H
#define _SCHEDEVT_H

/*

Definition of Event IDs for Keil MDK EventRecorder

*/
#include "EventRecorder.h"

#define EvtNodes 0x00   
#define EvtRing_User 0x01     
#define EvtRing_Int 0x02 
#define EvtRing_All 0x03 

/* Node events */

#define Evt_Sink         EventID (EventLevelAPI,   EvtNodes, 0x00)
#define Evt_SinkVal      EventID (EventLevelOp,   EvtNodes, 0x01)
#define Evt_Source       EventID (EventLevelAPI,   EvtNodes, 0x02)

/* User Ring Events */
#define Evt_UsrReserve      EventID (EventLevelOp,   EvtRing_User, 0x00)
#define Evt_UsrRelease      EventID (EventLevelOp,   EvtRing_User, 0x01)
#define Evt_UsrWait      EventID (EventLevelOp,   EvtRing_User, 0x02)
#define Evt_UsrFree      EventID (EventLevelOp,   EvtRing_User, 0x03)
#define Evt_UsrStatus      EventID (EventLevelDetail,   EvtRing_User, 0x04)


/* Interrupt Ring Events */
#define Evt_IntReserve      EventID (EventLevelOp,   EvtRing_Int, 0x00)
#define Evt_IntRelease      EventID (EventLevelOp,   EvtRing_Int, 0x01)
#define Evt_IntReleaseUser      EventID (EventLevelOp,   EvtRing_Int, 0x02)
#define Evt_IntStatus      EventID (EventLevelDetail,   EvtRing_Int, 0x03)


/* Other Ring Events */
#define Evt_Error      EventID (EventLevelError,   EvtRing_All, 0x00)



#endif