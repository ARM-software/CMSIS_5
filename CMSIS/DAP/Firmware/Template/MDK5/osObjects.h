/*
 * Copyright (c) 2013-2016 ARM Limited. All rights reserved.
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
 *
 * ----------------------------------------------------------------------
 *
 * $Date:        20. May 2015
 * $Revision:    V1.10
 *
 * Project:      CMSIS-DAP Template MDK5
 * Title:        osObjects.h CMSIS-DAP RTOS Objects
 *
 *---------------------------------------------------------------------------*/

#ifndef __osObjects_h__
#define __osObjects_h__

#include "cmsis_os.h"

#ifdef osObjectsExternal
extern osThreadId HID0_ThreadId;
#else
       osThreadId HID0_ThreadId;
#endif

extern void HID0_Thread (void const *arg);
osThreadDef(HID0_Thread, osPriorityNormal, 1U, 512U);

#endif  /* __osObjects_h__ */
