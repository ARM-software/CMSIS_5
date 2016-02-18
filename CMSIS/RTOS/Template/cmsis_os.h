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
 * $Date:        15. February 2016
 * $Revision:    V2.00
 *
 * Project:      CMSIS-RTOS API
 * Title:        cmsis_os.h template header file
 *
 * Version 0.02
 *    Initial Proposal Phase
 * Version 0.03
 *    osKernelStart added, optional feature: main started as thread
 *    osSemaphores have standard behavior
 *    osTimerCreate does not start the timer, added osTimerStart
 *    osThreadPass is renamed to osThreadYield
 * Version 1.01
 *    Support for C++ interface
 *     - const attribute removed from the osXxxxDef_t typedef's
 *     - const attribute added to the osXxxxDef macros
 *    Added: osTimerDelete, osMutexDelete, osSemaphoreDelete
 *    Added: osKernelInitialize
 * Version 1.02
 *    Control functions for short timeouts in microsecond resolution:
 *    Added: osKernelSysTick, osKernelSysTickFrequency, osKernelSysTickMicroSec
 *    Removed: osSignalGet 
 * Version 2.00
 *    Extended number of thread priorities
 *    Added: osXxxxDefInit macros for initializing osXxxxDef definitions
 *    Added: osXxxxDefM macros for defining objects with multiple instances
 *           (Mutex and Semaphore)
 *    Added: osXxxxExt macros for external reference to an object definition
 *    Added: osKernelTime, osKernelStop
 *    Added: osThreadState, osThreadGetState, osThreadSuspend, osThreadResume,
 *    Added: osTimerRunning
 *    Added: osPoolDelete
 *    Added: osMessageCount, osMessageReset, osMessageDelete
 *    Added: osMailCount, osMailReset, osMailDelete
 *    Added: osFlag object
 *---------------------------------------------------------------------------*/
 
#ifndef _CMSIS_OS_H
#define _CMSIS_OS_H
 
/// \note MUST REMAIN UNCHANGED: \b osCMSIS identifies the CMSIS-RTOS API version.
#define osCMSIS           0x20000U     ///< API version (main [31:16] .sub [15:0])
 
/// \note CAN BE CHANGED: \b osCMSIS_KERNEL identifies the underlying RTOS kernel and version number.
#define osCMSIS_KERNEL    0x10000U     ///< RTOS identification and version (main [31:16] .sub [15:0])
 
/// \note MUST REMAIN UNCHANGED: \b osKernelSystemId shall be consistent in every CMSIS-RTOS.
#define osKernelSystemId "KERNEL V1.00"   ///< RTOS identification string
 
/// \note MUST REMAIN UNCHANGED: \b osFeature_xxx shall be consistent in every CMSIS-RTOS.
#define osFeature_MainThread   1       ///< main thread      1=main can be thread, 0=not available
#define osFeature_Pool         1       ///< Memory Pools:    1=available, 0=not available
#define osFeature_MailQ        1       ///< Mail Queues:     1=available, 0=not available
#define osFeature_MessageQ     1       ///< Message Queues:  1=available, 0=not available
#define osFeature_Signals      8       ///< maximum number of Signal Flags available per thread
#define osFeature_Flags        16      ///< maximum number of Flag bits available
#define osFeature_Semaphore    30      ///< maximum count for \ref osSemaphoreCreate function
#define osFeature_Wait         1       ///< osWait function: 1=available, 0=not available
#define osFeature_SysTick      1       ///< osKernelSysTick functions: 1=available, 0=not available
 
#include <stdint.h>
#include <stddef.h>
 
#ifdef  __cplusplus
extern "C"
{
#endif
 
 
// ==== Enumeration, structures, defines ====
 
/// Priority used for thread control.
/// \note MUST REMAIN UNCHANGED: \b osPriority shall be consistent in every CMSIS-RTOS.
typedef enum  {
  osPriorityIdle          = -48,         ///< priority: idle (lowest)
  osPriorityIdle1,                       ///< priority: idle + 1
  osPriorityIdle2,                       ///< priority: idle + 2
  osPriorityIdle3,                       ///< priority: idle + 3
  osPriorityLow           = -32,         ///< priority: low
  osPriorityLow1,                        ///< priority: low + 1
  osPriorityLow2,                        ///< priority: low + 2
  osPriorityLow3,                        ///< priority: low + 3
  osPriorityBelowNormal   = -16,         ///< priority: below normal
  osPriorityBelowNormal1,                ///< priority: below normal + 1
  osPriorityBelowNormal2,                ///< priority: below normal + 2
  osPriorityBelowNormal3,                ///< priority: below normal + 3
  osPriorityNormal        =  0,          ///< priority: normal (default)
  osPriorityNormal1,                     ///< priority: normal + 1
  osPriorityNormal2,                     ///< priority: normal + 2
  osPriorityNormal3,                     ///< priority: normal + 3
  osPriorityAboveNormal   = +16,         ///< priority: above normal
  osPriorityAboveNormal1,                ///< priority: above normal + 1
  osPriorityAboveNormal2,                ///< priority: above normal + 2
  osPriorityAboveNormal3,                ///< priority: above normal + 3
  osPriorityHigh          = +32,         ///< priority: high
  osPriorityHigh1,                       ///< priority: high + 1
  osPriorityHigh2,                       ///< priority: high + 2
  osPriorityHigh3,                       ///< priority: high + 3
  osPriorityRealtime      = +48,         ///< priority: realtime
  osPriorityRealtime1,                   ///< priority: realtime + 1
  osPriorityRealtime2,                   ///< priority: realtime + 2
  osPriorityRealtime3,                   ///< priority: realtime + 3 (highest)
  osPriorityError         =  0x84,       ///< system cannot determine priority or thread has illegal priority
  os_priority_reserved    =  0x7FFFFFFF  ///< prevent from enum down-size compiler optimization.
} osPriority;
 
/// Timeout value.
/// \note MUST REMAIN UNCHANGED: \b osWaitForever shall be consistent in every CMSIS-RTOS.
#define osWaitForever     0xFFFFFFFFU    ///< wait forever timeout value
 
/// Flag attributes
/// \note MUST REMAIN UNCHANGED: \b osFlagWaitForAll and \b osFlagAutoClear shall be consistent in every CMSIS-RTOS.
#define osFlagWaitForAll  0x00000001U    ///< wait for all bits
#define osFlagAutoClear   0x00000002U    ///< clear active bits which have been specified to wait for
 
/// Status code values returned by CMSIS-RTOS functions.
/// \note MUST REMAIN UNCHANGED: \b osStatus shall be consistent in every CMSIS-RTOS.
typedef enum  {
  osOK                    =     0,       ///< function completed; no error or event occurred.
  osEventFlag             =  0x04,       ///< function completed; flag event occurred.
  osEventSignal           =  0x08,       ///< function completed; signal event occurred.
  osEventMessage          =  0x10,       ///< function completed; message event occurred.
  osEventMail             =  0x20,       ///< function completed; mail event occurred.
  osEventTimeout          =  0x40,       ///< function completed; timeout occurred.
  osErrorParameter        =  0x80,       ///< parameter error: a mandatory parameter was missing or specified an incorrect object.
  osErrorResource         =  0x81,       ///< resource not available: a specified resource was not available.
  osErrorTimeoutResource  =  0xC1,       ///< resource not available within given time: a specified resource was not available within the timeout period.
  osErrorISR              =  0x82,       ///< not allowed in ISR context: the function cannot be called from interrupt service routines.
  osErrorISRRecursive     =  0x83,       ///< function called multiple times from ISR with same object.
  osErrorPriority         =  0x84,       ///< system cannot determine priority or thread has illegal priority.
  osErrorNoMemory         =  0x85,       ///< system is out of memory: it was impossible to allocate or reserve memory for the operation.
  osErrorValue            =  0x86,       ///< value of a parameter is out of range.
  osErrorOS               =  0xFF,       ///< unspecified RTOS error: run-time error but no other error message fits.
  os_status_reserved      =  0x7FFFFFFF  ///< prevent from enum down-size compiler optimization.
} osStatus;
 
 
/// Thread State definitions.
/// \note MUST REMAIN UNCHANGED: \b osThreadState shall be consistent in every CMSIS-RTOS.
typedef enum {
  osThreadInactive          = 0x00,      ///< Inactive (not yet created).
  osThreadSuspended         = 0x01,      ///< Suspended.
  osThreadReady             = 0x02,      ///< Ready to Run.
  osThreadRunning           = 0x03,      ///< Running.
  osThreadWaiting           = 0x80,      ///< Waiting Mask.
  osThreadWaitingDelay      = 0x81,      ///< Waiting for Timeout (osDelay).
  osThreadWaitingEvent      = 0x82,      ///< Waiting for Event (osWait).
  osThreadWaitingFlag       = 0x83,      ///< Waiting for Flag (osFlagWait).
  osThreadWaitingSignal     = 0x84,      ///< Waiting for Signal (osSignalWait).
  osThreadWaitingSemaphore  = 0x85,      ///< Waiting for Semaphore (osSemaphoreWait).
  osThreadWaitingMutex      = 0x86,      ///< Waiting for Mutex (osMutexWait).
  osThreadWaitingMessageGet = 0x87,      ///< Waiting to receive Message (osMessageGet).
  osThreadWaitingMessagePut = 0x88,      ///< Waiting to send Message (osMessagePut).
  osThreadWaitingMailPut    = 0x89,      ///< Waiting to send Mail (osMailAlloc/osMailCAlloc).
  osThreadWaitingMailGet    = 0x8A,      ///< Waiting to receive Mail (osMailGet).
  osThreadError             = 0xFF,      ///< Error.
  os_thread_reserved        = 0x7FFFFFFF ///< prevent from enum down-size compiler optimization.
} osThreadState;

/// Timer type value for the timer definition.
/// \note MUST REMAIN UNCHANGED: \b os_timer_type shall be consistent in every CMSIS-RTOS.
typedef enum  {
  osTimerOnce             =     0,       ///< one-shot timer
  osTimerPeriodic         =     1        ///< repeating timer
} os_timer_type;
 
/// Entry point of a thread.
/// \note MUST REMAIN UNCHANGED: \b os_pthread shall be consistent in every CMSIS-RTOS.
typedef void (*os_pthread) (void const *argument);
 
/// Entry point of a timer call back function.
/// \note MUST REMAIN UNCHANGED: \b os_ptimer shall be consistent in every CMSIS-RTOS.
typedef void (*os_ptimer) (void const *argument);
 
// >>> the following data type definitions may shall adapted towards a specific RTOS
 
/// Thread ID identifies the thread (pointer to a thread control block).
/// \note CAN BE CHANGED: \b os_thread_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_thread_cb *osThreadId;
 
/// Timer ID identifies the timer (pointer to a timer control block).
/// \note CAN BE CHANGED: \b os_timer_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_timer_cb *osTimerId;
 
/// Flag ID identifies the flag (pointer to a flag control block).
/// \note CAN BE CHANGED: \b os_flag_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_flag_cb *osFlagId;
 
/// Mutex ID identifies the mutex (pointer to a mutex control block).
/// \note CAN BE CHANGED: \b os_mutex_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_mutex_cb *osMutexId;
 
/// Semaphore ID identifies the semaphore (pointer to a semaphore control block).
/// \note CAN BE CHANGED: \b os_semaphore_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_semaphore_cb *osSemaphoreId;
 
/// Pool ID identifies the memory pool (pointer to a memory pool control block).
/// \note CAN BE CHANGED: \b os_pool_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_pool_cb *osPoolId;
 
/// Message ID identifies the message queue (pointer to a message queue control block).
/// \note CAN BE CHANGED: \b os_messageQ_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_messageQ_cb *osMessageQId;
 
/// Mail ID identifies the mail queue (pointer to a mail queue control block).
/// \note CAN BE CHANGED: \b os_mailQ_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_mailQ_cb *osMailQId;
 
 
/// Thread Definition structure contains startup information of a thread.
/// \note CAN BE CHANGED: \b os_thread_def is implementation specific in every CMSIS-RTOS.
typedef struct os_thread_def  {
  os_pthread               pthread;    ///< start address of thread function
  osPriority             tpriority;    ///< initial thread priority
  uint32_t               instances;    ///< maximum number of instances of that thread function
  uint32_t               stacksize;    ///< stack size requirements in bytes; 0 is default stack size
} osThreadDef_t;
 
/// Timer Definition structure contains timer parameters.
/// \note CAN BE CHANGED: \b os_timer_def is implementation specific in every CMSIS-RTOS.
typedef struct os_timer_def  {
  os_ptimer                 ptimer;    ///< start address of a timer function
} osTimerDef_t;
 
/// Flag Definition structure contains setup information for a flag.
/// \note CAN BE CHANGED: \b os_flag_def is implementation specific in every CMSIS-RTOS.
typedef struct os_flag_def  {
  uint32_t                   dummy;    ///< dummy value.
} osFlagDef_t;
 
/// Mutex Definition structure contains setup information for a mutex.
/// \note CAN BE CHANGED: \b os_mutex_def is implementation specific in every CMSIS-RTOS.
typedef struct os_mutex_def  {
  uint32_t               instances;    ///< maximum number of instances
} osMutexDef_t;
 
/// Semaphore Definition structure contains setup information for a semaphore.
/// \note CAN BE CHANGED: \b os_semaphore_def is implementation specific in every CMSIS-RTOS.
typedef struct os_semaphore_def  {
  uint32_t               instances;    ///< maximum number of instances
} osSemaphoreDef_t;
 
/// Definition structure for memory block allocation.
/// \note CAN BE CHANGED: \b os_pool_def is implementation specific in every CMSIS-RTOS.
typedef struct os_pool_def  {
  uint32_t                 pool_sz;    ///< number of items (elements) in the pool
  uint32_t                 item_sz;    ///< size of an item
  void                       *pool;    ///< pointer to memory for pool
} osPoolDef_t;
 
/// Definition structure for message queue.
/// \note CAN BE CHANGED: \b os_messageQ_def is implementation specific in every CMSIS-RTOS.
typedef struct os_messageQ_def  {
  uint32_t                queue_sz;    ///< number of elements in the queue
  void                       *pool;    ///< memory array for messages
} osMessageQDef_t;
 
/// Definition structure for mail queue.
/// \note CAN BE CHANGED: \b os_mailQ_def is implementation specific in every CMSIS-RTOS.
typedef struct os_mailQ_def  {
  uint32_t                queue_sz;    ///< number of elements in the queue
  uint32_t                 item_sz;    ///< size of an item
  void                       *pool;    ///< memory array for mail
} osMailQDef_t;
 
/// Event structure contains detailed information about an event.
/// \note MUST REMAIN UNCHANGED: \b os_event shall be consistent in every CMSIS-RTOS.
///       However the struct may be extended at the end.
typedef struct  {
  osStatus                 status;     ///< status code: event or error information
  union  {
    uint32_t                    v;     ///< message as 32-bit value
    void                       *p;     ///< message or mail as void pointer
    int32_t               signals;     ///< signal flags
  } value;                             ///< event value
  union  {
    osMailQId             mail_id;     ///< mail id obtained by \ref osMailCreate
    osMessageQId       message_id;     ///< message id obtained by \ref osMessageCreate
  } def;                               ///< event definition
} osEvent;
 
 
//  ==== Kernel Control Functions ====
 
/// Initialize the RTOS Kernel for creating objects.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osKernelInitialize shall be consistent in every CMSIS-RTOS.
osStatus osKernelInitialize (void);
 
/// Start the RTOS Kernel.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osKernelStart shall be consistent in every CMSIS-RTOS.
osStatus osKernelStart (void);
 
/// Stop the RTOS Kernel.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osKernelStop shall be consistent in every CMSIS-RTOS.
osStatus osKernelStop (void);
 
/// Check if the RTOS kernel is already started.
/// \return 0 RTOS is not started, 1 RTOS is started.
/// \note MUST REMAIN UNCHANGED: \b osKernelRunning shall be consistent in every CMSIS-RTOS.
int32_t osKernelRunning (void);
 
/// Get the RTOS kernel time.
/// \return RTOS kernel current time in millisec (32-bit value).
/// \note MUST REMAIN UNCHANGED: \b osKernelTime shall be consistent in every CMSIS-RTOS.
uint32_t osKernelTime (void);
 
#if (defined (osFeature_SysTick)  &&  (osFeature_SysTick != 0))     // System Timer available
 
/// Get the RTOS kernel system timer counter 
/// \return RTOS kernel system timer as 32-bit value 
/// \note MUST REMAIN UNCHANGED: \b osKernelSysTick shall be consistent in every CMSIS-RTOS.
uint32_t osKernelSysTick (void);
 
/// The RTOS kernel system timer frequency in Hz
/// \note Reflects the system timer setting and is typically defined in a configuration file.
#define osKernelSysTickFrequency 100000000
 
/// Convert a microseconds value to a RTOS kernel system timer value.
/// \param         microsec     time value in microseconds.
/// \return time value normalized to the \ref osKernelSysTickFrequency
#define osKernelSysTickMicroSec(microsec) (((uint64_t)microsec * (osKernelSysTickFrequency)) / 1000000)
 
#endif    // System Timer available
 
//  ==== Thread Management ====
 
/// Create a Thread Definition with function, priority, and stack requirements.
/// \param         name          name of the thread function.
/// \param         priority      initial priority of the thread function.
/// \param         instances     number of possible thread instances.
/// \param         stacksz       stack size (in bytes) requirements for the thread function.
/// \note CAN BE CHANGED: The parameters to \b osThreadDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osThreadDef(name, priority, instances, stacksz)  \
extern const osThreadDef_t os_thread_def_##name
#else                            // define the object
#define osThreadDef(name, priority, instances, stacksz)  \
const osThreadDef_t os_thread_def_##name = \
{ (name), (priority), (instances), (stacksz) }
#endif
 
/// Thread definition initialization.
/// \param         thread        name of the thread function.
/// \param         priority      initial priority of the thread function.
/// \param         instances     number of possible thread instances.
/// \param         stacksz       stack size (in bytes) requirements for the thread function.
/// \note CAN BE CHANGED: The parameters to \b osThreadDefInit shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osThreadDefInit(thread, priority, instances, stacksz)  \
(osThreadDef_t){ (thread), (priority), (instances), (stacksz) }
 
/// External reference to a Thread definition.
/// \param         name          name of the thread definition object.
/// \note CAN BE CHANGED: The parameter to \b osThreadExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osThreadExt(name)  \
extern const osThreadDef_t os_thread_def_##name
 
/// Access a Thread definition.
/// \param         name          name of the thread definition object.
/// \note CAN BE CHANGED: The parameter to \b osThread shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osThread(name)  \
&os_thread_def_##name
 
/// Create a thread and add it to Active Threads and set it to state READY.
/// \param[in]     thread_def    thread definition referenced with \ref osThread.
/// \param[in]     argument      pointer that is passed to the thread function as start argument.
/// \return thread ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osThreadCreate shall be consistent in every CMSIS-RTOS.
osThreadId osThreadCreate (const osThreadDef_t *thread_def, void *argument);
 
/// Return the thread ID of the current running thread.
/// \return thread ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osThreadGetId shall be consistent in every CMSIS-RTOS.
osThreadId osThreadGetId (void);
 
/// Suspend execution of a thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadSuspend shall be consistent in every CMSIS-RTOS.
osStatus osThreadSuspend (osThreadId thread_id);
 
/// Resume execution of a thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadResume shall be consistent in every CMSIS-RTOS.
osStatus osThreadResume (osThreadId thread_id);
 
/// Terminate execution of a thread and remove it from Active Threads.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadTerminate shall be consistent in every CMSIS-RTOS.
osStatus osThreadTerminate (osThreadId thread_id);
 
/// Pass control to next thread that is in state \b READY.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadYield shall be consistent in every CMSIS-RTOS.
osStatus osThreadYield (void);
 
/// Change priority of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     priority      new priority value for the thread function.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadSetPriority shall be consistent in every CMSIS-RTOS.
osStatus osThreadSetPriority (osThreadId thread_id, osPriority priority);
 
/// Get current priority of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return current priority value of the specified thread.
/// \note MUST REMAIN UNCHANGED: \b osThreadGetPriority shall be consistent in every CMSIS-RTOS.
osPriority osThreadGetPriority (osThreadId thread_id);
 
/// Get current thread state.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return current thread state of the specified thread..
/// \note MUST REMAIN UNCHANGED: \b osThreadGetState shall be consistent in every CMSIS-RTOS.
osThreadState osThreadGetState (osThreadId thread_id);
 
 
//  ==== Generic Wait Functions ====
 
/// Wait for Timeout (Time Delay).
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue "time delay" value
/// \return status code that indicates the execution status of the function.
osStatus osDelay (uint32_t millisec);
 
#if (defined (osFeature_Wait)  &&  (osFeature_Wait != 0))     // Generic Wait available
 
/// Wait for Signal, Message, Mail, or Timeout.
/// \param[in] millisec          \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out
/// \return event that contains signal, message, or mail information or error code.
/// \note MUST REMAIN UNCHANGED: \b osWait shall be consistent in every CMSIS-RTOS.
osEvent osWait (uint32_t millisec);
 
#endif  // Generic Wait available
 
 
//  ==== Timer Management Functions ====
 
/// Define a Timer object.
/// \param         name          name of the timer object.
/// \param         function      name of the timer call back function.
/// \note CAN BE CHANGED: The parameter to \b osTimerDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osTimerDef(name, function)  \
extern const osTimerDef_t os_timer_def_##name
#else                            // define the object
#define osTimerDef(name, function)  \
const osTimerDef_t os_timer_def_##name = \
{ (function) }
#endif
 
/// Timer definition initialization.
/// \param         function      name of the timer call back function.
/// \note CAN BE CHANGED: The parameter to \b osTimerDefInit shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osTimerDefInit(function)  \
(osTimerDef_t){ (function) }
 
/// External reference to a Timer definition.
/// \param         name          name of the timer object.
/// \note CAN BE CHANGED: The parameter to \b osTimerExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osTimerExt(name)  \
extern const osTimerDef_t os_timer_def_##name
 
/// Access a Timer definition.
/// \param         name          name of the timer object.
/// \note CAN BE CHANGED: The parameter to \b osTimer shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osTimer(name) \
&os_timer_def_##name
 
/// Create a timer.
/// \param[in]     timer_def     timer object referenced with \ref osTimer.
/// \param[in]     type          osTimerOnce for one-shot or osTimerPeriodic for periodic behavior.
/// \param[in]     argument      argument to the timer call back function.
/// \return timer ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osTimerCreate shall be consistent in every CMSIS-RTOS.
osTimerId osTimerCreate (const osTimerDef_t *timer_def, os_timer_type type, void *argument);
 
/// Start or restart a timer.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue "time delay" value of the timer.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osTimerStart shall be consistent in every CMSIS-RTOS.
osStatus osTimerStart (osTimerId timer_id, uint32_t millisec);
 
/// Stop the timer.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osTimerStop shall be consistent in every CMSIS-RTOS.
osStatus osTimerStop (osTimerId timer_id);
 
/// Check if the timer is running.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerCreate.
/// \return 0 timer is not running, 1 timer is runing.
/// \note MUST REMAIN UNCHANGED: \b osTimerRunning shall be consistent in every CMSIS-RTOS.
int32_t osTimerRunning (osTimerId timer_id);
 
/// Delete a timer that was created by \ref osTimerCreate.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osTimerDelete shall be consistent in every CMSIS-RTOS.
osStatus osTimerDelete (osTimerId timer_id);
 
 
//  ==== Signal Management ====
 
/// Set the specified Signal Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     signals       specifies the signal flags of the thread that should be set.
/// \return previous signal flags of the specified thread or 0x80000000 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osSignalSet shall be consistent in every CMSIS-RTOS.
int32_t osSignalSet (osThreadId thread_id, int32_t signals);
 
/// Clear the specified Signal Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     signals       specifies the signal flags of the thread that shall be cleared.
/// \return previous signal flags of the specified thread or 0x80000000 in case of incorrect parameters or call from ISR.
/// \note MUST REMAIN UNCHANGED: \b osSignalClear shall be consistent in every CMSIS-RTOS.
int32_t osSignalClear (osThreadId thread_id, int32_t signals);
 
/// Wait for one or more Signal Flags to become signaled for the current \b RUNNING thread.
/// \param[in]     signals       wait until all specified signal flags set or 0 for any single signal flag.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return event flag information or error code.
/// \note MUST REMAIN UNCHANGED: \b osSignalWait shall be consistent in every CMSIS-RTOS.
osEvent osSignalWait (int32_t signals, uint32_t millisec);
 
 
//  ==== Flag Management ====
 
/// Define a Flag object.
/// \param         name          name of the flag object.
/// \note CAN BE CHANGED: The parameter to \b osFlagDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osFlagDef(name)  \
extern const osFlagDef_t os_flag_def_##name
#else                            // define the object
#define osFlagDef(name)  \
const osFlagDef_t os_flag_def_##name = { 0 }
#endif
 
/// External reference to a Flag definition.
/// \param         name          name of the flag object.
/// \note CAN BE CHANGED: The parameter to \b osFlagExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osFlagExt(name)  \
extern const osFlagDef_t os_flag_def_##name
 
/// Access a Flag definition.
/// \param         name          name of the flag object.
/// \note CAN BE CHANGED: The parameter to \b osFlag shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osFlag(name)  \
&os_flag_def_##name
 
/// Create and Initialize a Flag object.
/// \param[in]     flag_def      flag definition referenced with \ref osFlag.
/// \return flag ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osFlagCreate shall be consistent in every CMSIS-RTOS.
osFlagId osFlagCreate (const osFlagDef_t *flag_def);
 
/// Set the specified flags of a Flag object.
/// \param[in]     flag_id       flag ID obtained by \ref osFlagCreate.
/// \param[in]     flags         specifies the flags that shall be set.
/// \return flags of the specified Flag object after setting or 0x80000000 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osFlagSet shall be consistent in every CMSIS-RTOS.
int32_t osFlagSet (osFlagId flag_id, int32_t flags);
 
/// Clear the specified flags of a Flag object.
/// \param[in]     flag_id       flag ID obtained by \ref osFlagCreate.
/// \param[in]     flags         specifies the flags that shall be cleared.
/// \return flags of the specified Flag object before clearing or 0x80000000 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osFlagClear shall be consistent in every CMSIS-RTOS.
int32_t osFlagClear (osFlagId flag_id, int32_t flags);
 
/// Get the current flags of a Flag object.
/// \param[in]     flag_id       flag ID obtained by \ref osFlagCreate.
/// \return current flags of the specified Flag object or 0x80000000 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osFlagGet shall be consistent in every CMSIS-RTOS.
int32_t osFlagGet (osFlagId flag_id);
 
/// Wait for one or more Signal Flags to become signaled for the current \b RUNNING thread.
/// \param[in]     flag_id       flag ID obtained by \ref osFlagCreate.
/// \param[in]     flags         specifies the flags to wait for.
/// \param[in]     attributes    specifies optional attributes (osFlagWaitForAll, osFlagAutoClear).
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return flag information (before AutoClear) or error code.
/// \note MUST REMAIN UNCHANGED: \b osFlagWait shall be consistent in every CMSIS-RTOS.
osEvent osFlagWait (osFlagId flag_id, int32_t flags, uint32_t attributes, uint32_t millisec);
 
/// Delete a Flag object that was created by \ref osFlagCreate.
/// \param[in]     flag_id       flag ID obtained by \ref osFlagCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osFlagDelete shall be consistent in every CMSIS-RTOS.
osStatus osFlagDelete (osFlagId flag_id);
 
 
//  ==== Mutex Management ====
 
/// Define a Mutex.
/// \param         name          name of the mutex object.
/// \note CAN BE CHANGED: The parameter to \b osMutexDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osMutexDef(name)  \
extern const osMutexDef_t os_mutex_def_##name
#else                            // define the object
#define osMutexDef(name)  \
const osMutexDef_t os_mutex_def_##name = { 1U }
#endif
 
/// Define a Mutex with multiple instances.
/// \param         name          name of the mutex object.
/// \param         instances     number of possible instances.
/// \note CAN BE CHANGED: The parameter to \b osMutexDefM shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osMutexDefM(name, instances)  \
extern const osMutexDef_t os_mutex_def_##name
#else                            // define the object
#define osMutexDefM(name, instances)  \
const osMutexDef_t os_mutex_def_##name = { (instances) }
#endif
 
/// External reference to a Mutex definition.
/// \param         name          name of the mutex object.
/// \note CAN BE CHANGED: The parameter to \b osMutexExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMutexExt(name)  \
extern const osMutexDef_t os_mutex_def_##name
 
/// Access a Mutex definition.
/// \param         name          name of the mutex object.
/// \note CAN BE CHANGED: The parameter to \b osMutex shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMutex(name)  \
&os_mutex_def_##name
 
/// Create and Initialize a Mutex object.
/// \param[in]     mutex_def     mutex definition referenced with \ref osMutex.
/// \return mutex ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMutexCreate shall be consistent in every CMSIS-RTOS.
osMutexId osMutexCreate (const osMutexDef_t *mutex_def);
 
/// Wait until a Mutex becomes available.
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMutexWait shall be consistent in every CMSIS-RTOS.
osStatus osMutexWait (osMutexId mutex_id, uint32_t millisec);
 
/// Release a Mutex that was obtained by \ref osMutexWait.
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMutexRelease shall be consistent in every CMSIS-RTOS.
osStatus osMutexRelease (osMutexId mutex_id);
 
/// Delete a Mutex that was created by \ref osMutexCreate.
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMutexDelete shall be consistent in every CMSIS-RTOS.
osStatus osMutexDelete (osMutexId mutex_id);
 
 
//  ==== Semaphore Management Functions ====
 
#if (defined (osFeature_Semaphore)  &&  (osFeature_Semaphore != 0))     // Semaphore available
 
/// Define a Semaphore object.
/// \param         name          name of the semaphore object.
/// \note CAN BE CHANGED: The parameter to \b osSemaphoreDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osSemaphoreDef(name)  \
extern const osSemaphoreDef_t os_semaphore_def_##name
#else                            // define the object
#define osSemaphoreDef(name)  \
const osSemaphoreDef_t os_semaphore_def_##name = { 1U }
#endif
 
/// Define a Semaphore object with multiple instances.
/// \param         name          name of the semaphore object.
/// \param         instances     number of possible instances.
/// \note CAN BE CHANGED: The parameter to \b osSemaphoreDefM shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osSemaphoreDefM(name, instances)  \
extern const osSemaphoreDef_t os_semaphore_def_##name
#else                            // define the object
#define osSemaphoreDefM(name, instances)  \
const osSemaphoreDef_t os_semaphore_def_##name = { (instances) }
#endif
 
/// External reference to a Semaphore definition.
/// \param         name          name of the semaphore object.
/// \note CAN BE CHANGED: The parameter to \b osSemaphoreExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osSemaphoreExt(name)  \
extern const osSemaphoreDef_t os_semaphore_def_##name
 
/// Access a Semaphore definition.
/// \param         name          name of the semaphore object.
/// \note CAN BE CHANGED: The parameter to \b osSemaphore shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osSemaphore(name)  \
&os_semaphore_def_##name
 
/// Create and Initialize a Semaphore object used for managing resources.
/// \param[in]     semaphore_def semaphore definition referenced with \ref osSemaphore.
/// \param[in]     count         number of available resources.
/// \return semaphore ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osSemaphoreCreate shall be consistent in every CMSIS-RTOS.
osSemaphoreId osSemaphoreCreate (const osSemaphoreDef_t *semaphore_def, int32_t count);
 
/// Wait until a Semaphore token becomes available.
/// \param[in]     semaphore_id  semaphore object referenced with \ref osSemaphoreCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return number of available tokens, or -1 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osSemaphoreWait shall be consistent in every CMSIS-RTOS.
int32_t osSemaphoreWait (osSemaphoreId semaphore_id, uint32_t millisec);
 
/// Release a Semaphore token.
/// \param[in]     semaphore_id  semaphore object referenced with \ref osSemaphoreCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osSemaphoreRelease shall be consistent in every CMSIS-RTOS.
osStatus osSemaphoreRelease (osSemaphoreId semaphore_id);
 
/// Delete a Semaphore that was created by \ref osSemaphoreCreate.
/// \param[in]     semaphore_id  semaphore object referenced with \ref osSemaphoreCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osSemaphoreDelete shall be consistent in every CMSIS-RTOS.
osStatus osSemaphoreDelete (osSemaphoreId semaphore_id);
 
#endif     // Semaphore available
 
 
//  ==== Memory Pool Management Functions ====
 
#if (defined (osFeature_Pool)  &&  (osFeature_Pool != 0))  // Memory Pool Management available
 
/// \brief Define a Memory Pool.
/// \param         name          name of the memory pool.
/// \param         no            maximum number of blocks (objects) in the memory pool.
/// \param         type          data type of a single block (object).
/// \note CAN BE CHANGED: The parameter to \b osPoolDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osPoolDef(name, no, type)   \
extern const osPoolDef_t os_pool_def_##name
#else                            // define the object
#define osPoolDef(name, no, type)   \
const osPoolDef_t os_pool_def_##name = \
{ (no), sizeof(type), NULL }
#endif
 
/// Memory Pool definition initialization.
/// \param         no            maximum number of blocks (objects) in the memory pool.
/// \param         type          data type of a single block (object).
/// \note CAN BE CHANGED: The parameter to \b osPoolDefInit shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osPoolDefInit(no, type)  \
(osPoolDef_t){ (no), sizeof(type), NULL }
 
/// External reference to a Memory Pool definition.
/// \param         name          name of the memory pool.
/// \note CAN BE CHANGED: The parameter to \b osPoolExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osPoolExt(name)  \
extern const osPoolDef_t os_pool_def_##name
 
/// \brief Access a Memory Pool definition.
/// \param         name          name of the memory pool.
/// \note CAN BE CHANGED: The parameter to \b osPool shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osPool(name) \
&os_pool_def_##name
 
/// Create and Initialize a memory pool.
/// \param[in]     pool_def      memory pool definition referenced with \ref osPool.
/// \return memory pool ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osPoolCreate shall be consistent in every CMSIS-RTOS.
osPoolId osPoolCreate (const osPoolDef_t *pool_def);
 
/// Allocate a memory block from a memory pool.
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \return address of the allocated memory block or NULL in case of no memory available.
/// \note MUST REMAIN UNCHANGED: \b osPoolAlloc shall be consistent in every CMSIS-RTOS.
void *osPoolAlloc (osPoolId pool_id);
 
/// Allocate a memory block from a memory pool and set memory block to zero.
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \return address of the allocated memory block or NULL in case of no memory available.
/// \note MUST REMAIN UNCHANGED: \b osPoolCAlloc shall be consistent in every CMSIS-RTOS.
void *osPoolCAlloc (osPoolId pool_id);
 
/// Return an allocated memory block back to a specific memory pool.
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \param[in]     block         address of the allocated memory block that is returned to the memory pool.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osPoolFree shall be consistent in every CMSIS-RTOS.
osStatus osPoolFree (osPoolId pool_id, void *block);
 
/// Delete a memory pool that was created by \ref osPoolCreate.
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osPoolDelete shall be consistent in every CMSIS-RTOS.
osStatus osPoolDelete (osPoolId pool_id);
 
#endif   // Memory Pool Management available
 
 
//  ==== Message Queue Management Functions ====
 
#if (defined (osFeature_MessageQ)  &&  (osFeature_MessageQ != 0))     // Message Queues available
 
/// \brief Create a Message Queue Definition.
/// \param         name          name of the queue.
/// \param         queue_sz      maximum number of messages in the queue.
/// \param         type          data type of a single message element (for debugger).
/// \note CAN BE CHANGED: The parameter to \b osMessageQDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osMessageQDef(name, queue_sz, type)   \
extern const osMessageQDef_t os_messageQ_def_##name
#else                            // define the object
#define osMessageQDef(name, queue_sz, type)   \
const osMessageQDef_t os_messageQ_def_##name = \
{ (queue_sz), NULL }
#endif
 
/// Message Queue definition initialization.
/// \param         queue_sz      maximum number of messages in the queue.
/// \param         type          data type of a single message element (for debugger).
/// \note CAN BE CHANGED: The parameter to \b osMessageQDefInit shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMessageQDefInit(queue_sz, type)  \
(osMessageQDef_t){ (queue_sz), NULL }
 
/// External reference to a Message Queue definition.
/// \note CAN BE CHANGED: The parameter to \b osMessageQExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
/// \param         name          name of the queue.
#define osMessageQExt(name)  \
extern const osMessageQDef_t os_messageQ_def_##name
 
/// \brief Access a Message Queue definition.
/// \param         name          name of the queue.
/// \note CAN BE CHANGED: The parameter to \b osMessageQ shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMessageQ(name) \
&os_messageQ_def_##name
 
/// Create and Initialize a Message Queue.
/// \param[in]     queue_def     queue definition referenced with \ref osMessageQ.
/// \param[in]     thread_id     thread ID (obtained by \ref osThreadCreate or \ref osThreadGetId) or NULL.
/// \return message queue ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMessageCreate shall be consistent in every CMSIS-RTOS.
osMessageQId osMessageCreate (const osMessageQDef_t *queue_def, osThreadId thread_id);
 
/// Put a Message to a Queue.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \param[in]     info          message information.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMessagePut shall be consistent in every CMSIS-RTOS.
osStatus osMessagePut (osMessageQId queue_id, uint32_t info, uint32_t millisec);
 
/// Get a Message or Wait for a Message from a Queue.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return event information that includes status code.
/// \note MUST REMAIN UNCHANGED: \b osMessageGet shall be consistent in every CMSIS-RTOS.
osEvent osMessageGet (osMessageQId queue_id, uint32_t millisec);
 
/// Get the number of messages in the Queue.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \return number of messages currently queued.
/// \note MUST REMAIN UNCHANGED: \b osMessageCount shall be consistent in every CMSIS-RTOS.
uint32_t osMessageCount (osMessageQId queue_id);
 
/// Reset a Message Queue to original empty state.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMessageReset shall be consistent in every CMSIS-RTOS.
osStatus osMessageReset (osMessageQId queue_id);
 
/// Delete a Message Queue that was created by \ref osMessageCreate.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMessageDelete shall be consistent in every CMSIS-RTOS.
osStatus osMessageDelete (osMessageQId queue_id);
 
#endif     // Message Queues available
 
 
//  ==== Mail Queue Management Functions ====
 
#if (defined (osFeature_MailQ)  &&  (osFeature_MailQ != 0))     // Mail Queues available
 
/// \brief Create a Mail Queue Definition.
/// \param         name          name of the queue
/// \param         queue_sz      maximum number of messages in queue
/// \param         type          data type of a single message element
/// \note CAN BE CHANGED: The parameter to \b osMailQDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osMailQDef(name, queue_sz, type) \
extern const osMailQDef_t os_mailQ_def_##name
#else                            // define the object
#define osMailQDef(name, queue_sz, type) \
const osMailQDef_t os_mailQ_def_##name =  \
{ (queue_sz), sizeof(type), NULL }
#endif
 
/// Mail Queue definition initialization.
/// \param         queue_sz      maximum number of messages in queue
/// \param         type          data type of a single message element
/// \note CAN BE CHANGED: The parameter to \b osMailQDefInit shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMailQDefInit(queue_sz, type)  \
(osMailQDef_t){ (queue_sz), sizeof(type), NULL }

/// External reference to a Mail Queue definition.
/// \param         name          name of the queue.
/// \note CAN BE CHANGED: The parameter to \b osMailQExt shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMailQExt(name)  \
extern const osMailQDef_t os_mailQ_def_##name

/// \brief Access a Mail Queue definition.
/// \param         name          name of the queue.
/// \note CAN BE CHANGED: The parameter to \b osMailQ shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMailQ(name)  \
&os_mailQ_def_##name
 
/// Create and Initialize mail queue.
/// \param[in]     queue_def     reference to the mail queue definition obtain with \ref osMailQ
/// \param[in]     thread_id     thread ID (obtained by \ref osThreadCreate or \ref osThreadGetId) or NULL.
/// \return mail queue ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMailCreate shall be consistent in every CMSIS-RTOS.
osMailQId osMailCreate (const osMailQDef_t *queue_def, osThreadId thread_id);
 
/// Allocate a memory block from a mail.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out
/// \return pointer to memory block that can be filled with mail or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMailAlloc shall be consistent in every CMSIS-RTOS.
void *osMailAlloc (osMailQId queue_id, uint32_t millisec);
 
/// Allocate a memory block from a mail and set memory block to zero.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out
/// \return pointer to memory block that can be filled with mail or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMailCAlloc shall be consistent in every CMSIS-RTOS.
void *osMailCAlloc (osMailQId queue_id, uint32_t millisec);
 
/// Put a mail to a queue.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     mail          memory block previously allocated with \ref osMailAlloc or \ref osMailCAlloc.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMailPut shall be consistent in every CMSIS-RTOS.
osStatus osMailPut (osMailQId queue_id, void *mail);
 
/// Get a mail from a queue.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out
/// \return event that contains mail information or error code.
/// \note MUST REMAIN UNCHANGED: \b osMailGet shall be consistent in every CMSIS-RTOS.
osEvent osMailGet (osMailQId queue_id, uint32_t millisec);
 
/// Free a memory block from a mail.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     mail          pointer to the memory block that was obtained with \ref osMailGet.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMailFree shall be consistent in every CMSIS-RTOS.
osStatus osMailFree (osMailQId queue_id, void *mail);
 
/// Get the number of mails in the Queue.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \return number of mails currently queued.
/// \note MUST REMAIN UNCHANGED: \b osMailCount shall be consistent in every CMSIS-RTOS.
uint32_t osMailCount (osMailQId queue_id);
 
/// Reset a Mail Queue to original empty state.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMailReset shall be consistent in every CMSIS-RTOS.
osStatus osMailReset (osMailQId queue_id);
 
/// Delete a Mail Queue that was created by \ref osMailCreate.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMailDelete shall be consistent in every CMSIS-RTOS.
osStatus osMailDelete (osMailQId queue_id);
 
#endif  // Mail Queues available
 
 
#ifdef  __cplusplus
}
#endif
 
#endif  // _CMSIS_OS_H
