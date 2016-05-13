/*
 * Copyright (c) 2013-2016 ARM Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ----------------------------------------------------------------------
 *
 * $Date:        7. April 2016
 * $Revision:    V1.0
 *
 * Project:      CMSIS-RTOS API
 * Title:        cmsis_os_v1.c V1 module file
 *---------------------------------------------------------------------------*/

#include <string.h>
#include "cmsis_os.h"

#ifdef  osCMSIS_API_V1

// Kernel
int32_t osKernelRunning (void) {
  return ((osKernelGetState() == osKernelRunning_) ? 1 : 0);
}

// Thread
osThreadId osThreadCreate (const osThreadDef_t *thread_def, void *argument) {
  osThreadAttr_t attr;

  memset(&attr, 0, sizeof(attr));
  attr.priority   = thread_def->tpriority;
  attr.stack_size = thread_def->stacksize;

  return osThreadNew(thread_def->pthread, argument, &attr);
}


// Wait

#if (defined(osFeature_Wait) && (osFeature_Wait != 0))

osEvent osWait (uint32_t millisec) {
  osEvent  event;
  uint32_t event_bits;

  memset(&event, 0, sizeof(event));
  event_bits = osEventWait(millisec);

  if (event_bits & osEventThreadFlags) {
    event.status |= osEventSignal;
  }
  if (event_bits & osEventMessageQueue) {
    event.status |= osEventMessage;
  }
  if (event_bits & osEventMailQueue) {
    event.status |= osEventMail;
  }
  return event;  // Only event.status is returned!
}

#endif  // Wait


// Signals

int32_t osSignalSet (osThreadId thread_id, int32_t signals) {
  osStatus status;
  int32_t  flags;

  flags = osThreadFlagsGet(thread_id);
  if (flags < 0) {
    return (int32_t)0x80000000U;
  }
  status = osThreadFlagsSet(thread_id, signals);
  if (status != osOK) {
    return (int32_t)0x80000000U;
  }
  return flags;
}

int32_t osSignalClear (osThreadId thread_id, int32_t signals) {
  osStatus status;
  int32_t  flags;

  flags = osThreadFlagsGet(thread_id);
  if (flags < 0) {
    return (int32_t)0x80000000U;
  }
  status = osThreadFlagsClear(thread_id, signals);
  if (status != osOK) {
    return (int32_t)0x80000000U;
  }
  return flags;
}

#define ThreadFlagsMask (int32_t)((1U<<osFeature_ThreadFlags)-1U)

osEvent osSignalWait (int32_t signals, uint32_t millisec) {
  osEvent event;
  int32_t flags;

  if (signals != 0) {
    flags = osThreadFlagsWait(signals,         osFlagsWaitAll | osFlagsAutoClear, millisec);
  } else {
    flags = osThreadFlagsWait(ThreadFlagsMask, osFlagsWaitAny | osFlagsAutoClear, millisec);
  }
  if (flags >= 0) {
    event.status = osEventSignal;
    event.value.signals = flags;
  } else {
    switch (flags) {
      case osErrorResource:
        event.status = osOK;
        break;
      case osErrorTimeout:
        event.status = osEventTimeout;
        break;
      case osErrorParameter:
        event.status = osErrorValue;
        break;
      default:
        event.status = (osStatus)flags;
        break;
    }
  }
  return event;
}


// Timer
osTimerId osTimerCreate (const osTimerDef_t *timer_def, os_timer_type type, void *argument) {
  return osTimerNew(timer_def->ptimer, type, argument, NULL);
}

// Mutex
osMutexId osMutexCreate (const osMutexDef_t *mutex_def) {
  osMutexAttr_t attr = osMutexAttrInit("Mutex", osMutexRecursive);
  (void)mutex_def;

  return osMutexNew(&attr);
}


// Semaphore

osSemaphoreId osSemaphoreCreate (const osSemaphoreDef_t *semaphore_def, int32_t count) {
  (void)semaphore_def;

  return osSemaphoreNew((uint32_t)count, (uint32_t)count, NULL);
}

int32_t osSemaphoreWait (osSemaphoreId semaphore_id, uint32_t millisec) {
  osStatus status;

  status = osSemaphoreAcquire(semaphore_id, millisec);
  if (status == osOK) {
    return 1;
  }
  if ((status == osErrorResource) || (status == osErrorTimeout)) {
    return 0;
  }
  return -1;
}


// Memory Pool

#if (defined(osFeature_Pool) && (osFeature_Pool != 0))

osPoolId osPoolCreate (const osPoolDef_t *pool_def) {
  return osMemoryPoolNew(pool_def->pool_sz, pool_def->item_sz, pool_def->pool, NULL);
}

void *osPoolCAlloc (osPoolId pool_id) {
  osStatus status;
  void    *block;
  uint32_t block_size;

  status = osMemoryPoolGetInfo(pool_id, NULL, &block_size, NULL);
  if (status != osOK) {
    return NULL;
  }
  block = osMemoryPoolAlloc(pool_id);
  if (block != NULL) {
    memset(block, 0, block_size);
  }
  return block;
}

#endif  // Memory Pool


// Message Queue

#if (defined(osFeature_MessageQ) && (osFeature_MessageQ != 0))

osMessageQId osMessageCreate (const osMessageQDef_t *queue_def, osThreadId thread_id) {
  osMessageQueueAttr_t attr;
  
  memset(&attr, 0, sizeof(attr));
  attr.thread_id = thread_id;

  return osMessageQueueNew(queue_def->queue_sz, queue_def->pool, &attr);
}

osEvent osMessageGet (osMessageQId queue_id, uint32_t millisec) {
  osStatus status;
  osEvent  event;
  uint32_t message;

  status = osMessageQueueGet(queue_id, &message, millisec);
  if (status == osOK) {
    event.status = osEventMessage;
    event.value.v = message;
  } else {
    switch (status) {
      case osErrorResource:
        event.status = osOK;
        break;
      case osErrorTimeout:
        event.status = osEventTimeout;
        break;
      default:
        event.status = status;
        break;
    }
  }
  return event;
}

#endif  // Message Queue


// Mail Queue

#if (defined(osFeature_MailQ) && (osFeature_MailQ != 0))

osMailQId osMailCreate (const osMailQDef_t *queue_def, osThreadId thread_id) {
  osMailQueueAttr_t attr;

  memset(&attr, 0, sizeof(attr));
  attr.thread_id = thread_id;

  return osMailQueueNew(queue_def->queue_sz, queue_def->item_sz, queue_def->pool, &attr);
}

void *osMailCAlloc (osMailQId queue_id, uint32_t millisec) {
  osStatus status;
  void    *mail;
  uint32_t mail_size;

  status = osMailQueueGetInfo(queue_id, NULL, &mail_size, NULL);
  if (status != osOK) {
    return NULL;
  }
  mail = osMailQueueAlloc(queue_id, millisec);
  if (mail != NULL) {
    memset(mail, 0, mail_size);
  }
  return mail;
}

osEvent osMailGet (osMailQId queue_id, uint32_t millisec) {
  osStatus status;
  osEvent  event;
  void    *mail;

  status = osMailQueueGet(queue_id, &mail, millisec);
  if (status == osOK) {
    event.status = osEventMail;
    event.value.p = mail;
  } else {
    switch (status) {
      case osErrorResource:
        event.status = osOK;
        break;
      case osErrorTimeout:
        event.status = osEventTimeout;
        break;
      default:
        event.status = status;
        break;
    }
  }
  return event;
}

#endif  // Mail Queue


#endif  // osCMSIS_API_V1
