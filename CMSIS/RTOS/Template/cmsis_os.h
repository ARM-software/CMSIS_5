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
 * $Revision:    V2.0
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
 * Version 2.0
 *    OS object's resources dynamically allocated rather than statically:
 *     - added: osXxxxNew functions which replace osXxxxCreate
 *     - added: osXxxxAttr_t structures and osXxxxAttrInit macros
 *     - removed: osXxxxCreate functions, osXxxxDef_t structures
 *     - removed: osXxxxDef and osXxxx macros
 *    osStatus codes simplified
 *    osEvent return structure removed
 *    Kernel:
 *     - added: osKernelState and osKernelGetState (replaces osKernelRunning)
 *     - added: osKernelSuspend, osKernelResume
 *     - added: osKernelTime
 *    Thread:
 *     - extended number of thread priorities
 *     - replaced osThreadCreate with osThreadNew
 *     - added: osThreadState and osThreadGetState
 *     - added: osThreadSuspend, osThreadResume
 *     - added: Thread Flags (moved from Signals) 
 *    Signals:
 *     - renamed osSignals to osThreadFlags (moved to Thread Flags)
 *     - changed return value of Set/Clear/Wait functions
 *     - extended Wait function (options)
 *     - added osThreadFlagsGet
 *    Event Flags:
 *     - added new independent object for handling Event Flags
 *    Delay and Wait functions:
 *     - added osDelayUntil
 *     - replaced osWait with osEventWait (changed return value)
 *    Timer:
 *     - replaced osTimerCreate with osTimerNew
 *     - added osTimerIsRunning
 *    Mutex:
 *     - extended: Recursive or Non-Recursive Mutex (default)
 *     - replaced osMutexCreate with osMutexNew
 *     - renamed osMutexWait to osMutexAcquire
 *    Semaphore:
 *     - extended: maximum and initial token count
 *     - replaced osSemaphoreCreate with osSemaphoreNew
 *     - renamed osSemaphoreWait to osSemaphoreAcquire (changed return value)
 *    Memory Pool:
 *     - using osMemoryPool prefix instead of osPool
 *     - replaced osPoolCreate with osMemoryPoolNew
 *     - added: osMemoryPoolGetInfo osMemoryPoolDelete
 *     - removed: osPoolCAlloc
 *    Message Queue:
 *     - using osMessageQueue prefix instead of osMessage
 *     - replaced osMessageCreate with osMessageQueueNew
 *     - changed return value of osMessageQueueGet
 *     - added: osMessageQueueGetInfo, osMessageQueueReset, osMessageQueueDelete
 *    Mail Queue:
 *     - using osMailQueue prefix instead of osMail
 *     - replaced osMailCreate with osMailQueueNew
 *     - changed return value of osMailQueueGet
 *     - added: osMailQueueGetInfo, osMailQueueReset, osMailQueueDelete
 *     - removed: osMailCAlloc
 *---------------------------------------------------------------------------*/
 
#ifndef _CMSIS_OS_H
#define _CMSIS_OS_H
 
/// \b osCMSIS identifies the CMSIS-RTOS API version.
#define osCMSIS             0x20000U     ///< API version (main [31:16] .sub [15:0])
 
/// \note CAN BE CHANGED: \b osCMSIS_KERNEL identifies the underlying RTOS kernel and version number.
#define osCMSIS_KERNEL      0x10000U     ///< RTOS identification and version (main [31:16] .sub [15:0])
 
/// \b osKernelSystemId identifies the underlying RTOS kernel.
#define osKernelSystemId "KERNEL V1.0"   ///< RTOS identification string
 
/// \b osFeature_xxx identifies RTOS features.
#define osFeature_ThreadFlags     31U    ///< number of Thread Flags available per thread (max=31, min=8)
#define osFeature_EventFlags      31U    ///< number of Event Flags available per object  (max=31, min=0)
#define osFeature_SemaphoreTokens 65535U ///< maximum number of tokens per semaphore (min=1)
#define osFeature_KernelSysTick   1      ///< osKernelSysTick: 1=available, 0=not available
#define osFeature_EventWait       1      ///< osEventWait:     1=available, 0=not available
#define osFeature_MemoryPool      1      ///< Memory Pools:    1=available, 0=not available
#define osFeature_MessageQueue    1      ///< Message Queues:  1=available, 0=not available
#define osFeature_MailQueue       1      ///< Mail Queues:     1=available, 0=not available
 
#ifdef  osCMSIS_API_V1
#define osFeature_MainThread      0      ///< main is not a thread
#define osFeature_Signals         osFeature_ThreadFlags
#define osFeature_Semaphore       osFeature_SemaphoreTokens
#define osFeature_SysTick         osFeature_KernelSysTick
#define osFeature_Wait            osFeature_EventWait
#define osFeature_Pool            osFeature_MemoryPool
#define osFeature_MessageQ        osFeature_MessageQueue
#define osFeature_MailQ           osFeature_MailQueue
#endif

#include <stdint.h>
#include <stddef.h>
 
#ifdef  __cplusplus
extern "C"
{
#endif
 
 
// ==== Enumerations, structures, defines ====
 
/// Priority values.
typedef enum {
  osPriorityNone          =  0,          ///< No priority (not initialized).
  osPriorityIdle          =  1,          ///< Reserved for Idle thread.
  osPriorityLow           =  8,          ///< Priority: low
  osPriorityLow1          =  8+1,        ///< Priority: low + 1
  osPriorityLow2          =  8+2,        ///< Priority: low + 2
  osPriorityLow3          =  8+3,        ///< Priority: low + 3
  osPriorityLow4          =  8+4,        ///< Priority: low + 4
  osPriorityLow5          =  8+5,        ///< Priority: low + 5
  osPriorityLow6          =  8+6,        ///< Priority: low + 6
  osPriorityLow7          =  8+7,        ///< Priority: low + 7
  osPriorityBelowNormal   = 16,          ///< Priority: below normal
  osPriorityBelowNormal1  = 16+1,        ///< Priority: below normal + 1
  osPriorityBelowNormal2  = 16+2,        ///< Priority: below normal + 2
  osPriorityBelowNormal3  = 16+3,        ///< Priority: below normal + 3
  osPriorityBelowNormal4  = 16+4,        ///< Priority: below normal + 4
  osPriorityBelowNormal5  = 16+5,        ///< Priority: below normal + 5
  osPriorityBelowNormal6  = 16+6,        ///< Priority: below normal + 6
  osPriorityBelowNormal7  = 16+7,        ///< Priority: below normal + 7
  osPriorityNormal        = 24,          ///< Priority: normal
  osPriorityNormal1       = 24+1,        ///< Priority: normal + 1
  osPriorityNormal2       = 24+2,        ///< Priority: normal + 2
  osPriorityNormal3       = 24+3,        ///< Priority: normal + 3
  osPriorityNormal4       = 24+4,        ///< Priority: normal + 4
  osPriorityNormal5       = 24+5,        ///< Priority: normal + 5
  osPriorityNormal6       = 24+6,        ///< Priority: normal + 6
  osPriorityNormal7       = 24+7,        ///< Priority: normal + 7
  osPriorityAboveNormal   = 32,          ///< Priority: above normal
  osPriorityAboveNormal1  = 32+1,        ///< Priority: above normal + 1
  osPriorityAboveNormal2  = 32+2,        ///< Priority: above normal + 2
  osPriorityAboveNormal3  = 32+3,        ///< Priority: above normal + 3
  osPriorityAboveNormal4  = 32+4,        ///< Priority: above normal + 4
  osPriorityAboveNormal5  = 32+5,        ///< Priority: above normal + 5
  osPriorityAboveNormal6  = 32+6,        ///< Priority: above normal + 6
  osPriorityAboveNormal7  = 32+7,        ///< Priority: above normal + 7
  osPriorityHigh          = 40,          ///< Priority: high
  osPriorityHigh1         = 40+1,        ///< Priority: high + 1
  osPriorityHigh2         = 40+2,        ///< Priority: high + 2
  osPriorityHigh3         = 40+3,        ///< Priority: high + 3
  osPriorityHigh4         = 40+4,        ///< Priority: high + 4
  osPriorityHigh5         = 40+5,        ///< Priority: high + 5
  osPriorityHigh6         = 40+6,        ///< Priority: high + 6
  osPriorityHigh7         = 40+7,        ///< Priority: high + 7
  osPriorityRealtime      = 48,          ///< Priority: realtime
  osPriorityRealtime1     = 48+1,        ///< Priority: realtime + 1
  osPriorityRealtime2     = 48+2,        ///< Priority: realtime + 2
  osPriorityRealtime3     = 48+3,        ///< Priority: realtime + 3
  osPriorityRealtime4     = 48+4,        ///< Priority: realtime + 4
  osPriorityRealtime5     = 48+5,        ///< Priority: realtime + 5
  osPriorityRealtime6     = 48+6,        ///< Priority: realtime + 6
  osPriorityRealtime7     = 48+7,        ///< Priority: realtime + 7
  osPriorityISR           = 56,          ///< Reserved for ISR deferred thread.
  osPriorityError         = -1,          ///< System cannot determine priority or illegal priority.
  os_priority_reserved    = 0x7FFFFFFF   ///< Prevents enum down-size compiler optimization.
} osPriority;
 
/// Kernel state.
typedef enum {
  osKernelInactive        =  0,          ///< Inactive.
#ifdef  osCMSIS_API_V1
  osKernelRunning_        =  1,          ///< Running.
#else
  osKernelRunning         =  1,          ///< Running.
#endif
  osKernelSuspended       =  2,          ///< Suspended.
  os_kernel_reserved      = 0x7FFFFFFFU  ///< Prevents enum down-size compiler optimization.
} osKernelState;
 
/// Thread state.
typedef enum {
  osThreadInactive        =  0,          ///< Inactive (not yet created).
  osThreadRunning         =  1,          ///< Running.
  osThreadReady           =  2,          ///< Ready to run.
  osThreadSuspended       =  3,          ///< Suspended.
  osThreadWaiting         =  4,          ///< Waiting.
  osThreadError           = -1,          ///< Error.
  os_thread_reserved      = 0x7FFFFFFF   ///< Prevents enum down-size compiler optimization.
} osThreadState;
 
/// Entry point of a thread.
typedef void (*os_pthread) (void const *argument);
 
/// Entry point of a timer call back function.
typedef void (*os_ptimer) (void const *argument);
 
/// Timer type.
typedef enum {
  osTimerOnce             = 0,           ///< One-shot timer.
  osTimerPeriodic         = 1            ///< Repeating timer.
} os_timer_type;
 
/// Timeout value.
#define osWaitForever       0xFFFFFFFFU  ///< Wait forever timeout value.
 
/// Flags options (\ref osThreadFlagsWait and \ref osEventFlagsWait).
#define osFlagsWaitAny      0x00000000U  ///< Wait for any flag (default).
#define osFlagsWaitAll      0x00000001U  ///< Wait for all flags.
#define osFlagsAutoClear    0x00000002U  ///< Clear flags which have been specified to wait for.
 
/// Mutex attributes (attr_bits in \ref osMutexAttr_t).
#define osMutexNonRecursive 0x00000000U  ///< Non-recursive mutex (default).
#define osMutexRecursive    0x00000001U  ///< Recursive mutex.
 
/// Event bit-mask (\ref osEventWait).
#define osEventNone         0x00000000U  ///< No event (timeout).
#define osEventThreadFlags  0x00000001U  ///< Thread Flags.
#define osEventEventFlags   0x00000002U  ///< Event Flags.
#define osEventMessageQueue 0x00000004U  ///< Message Queue not empty.
#define osEventMailQueue    0x00000008U  ///< Mail Queue not empty.
 
 
/// Status code values returned by CMSIS-RTOS functions.
#ifdef  osCMSIS_API_V1
typedef enum {
  osOK                    =  0,          ///< Function completed; no error or event occurred.
  osEventSignal           =  0x08,       ///< Function completed; signal event occurred.
  osEventMessage          =  0x10,       ///< Function completed; message event occurred.
  osEventMail             =  0x20,       ///< Function completed; mail event occurred.
  osEventTimeout          =  0x40,       ///< Function completed; timeout occurred.
  osErrorOS               = -1,          ///< Unspecified RTOS error: run-time error but no other error message fits.
  osErrorTimeoutResource  = -2,          ///< Resource not available within given time: a specified resource was not available within the timeout period.
  osErrorResource         = -3,          ///< Resource not available: a specified resource was not available.
  osErrorParameter        = -4,          ///< Parameter error: a mandatory parameter was missing or specified an incorrect object.
  osErrorNoMemory         = -5,          ///< System is out of memory: it was impossible to allocate or reserve memory for the operation.
  osErrorISR              = -6,          ///< Not allowed in ISR context: the function cannot be called from interrupt service routines.
  osErrorISRRecursive     = -7,          ///< Function called multiple times from ISR with same object.
  osErrorValue            = -127,        ///< Value of a parameter is out of range.
  osErrorPriority         = -128,        ///< System cannot determine priority or thread has illegal priority.
  os_status_reserved      =  0x7FFFFFFF  ///< Prevents enum down-size compiler optimization.
} osStatus;
#define osError              osErrorOS
#define osErrorTimeout       osErrorTimeoutResource
#define osErrorISR_Recursive osErrorISRRecursive
#else
typedef enum {
  osOK                    =  0,          ///< Operation completed successfully.
  osError                 = -1,          ///< Unspecified RTOS error: run-time error but no other error message fits.
  osErrorTimeout          = -2,          ///< Operation not completed within the timeout period.
  osErrorResource         = -3,          ///< Resource not available.
  osErrorParameter        = -4,          ///< Parameter error.
  osErrorNoMemory         = -5,          ///< System is out of memory: it was impossible to allocate or reserve memory for the operation.
  osErrorISR              = -6,          ///< Not allowed in ISR context: the function cannot be called from interrupt service routines.
  osErrorISR_Recursive    = -7,          ///< Function called multiple times from ISR with same object.
  os_status_reserved      = 0x7FFFFFFF   ///< Prevents enum down-size compiler optimization.
} osStatus;
#endif
 
 
// >>> the following data type definitions may be adapted towards a specific RTOS
 
/// \details Thread ID identifies the thread (pointer to a thread control block).
/// \note CAN BE CHANGED: \b os_thread_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_thread_cb *osThreadId;
 
/// \details Timer ID identifies the timer (pointer to a timer control block).
/// \note CAN BE CHANGED: \b os_timer_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_timer_cb *osTimerId;
 
/// \details Event Flags ID identifies the event flags (pointer to a event flags control block).
/// \note CAN BE CHANGED: \b os_event_flags_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_event_flags_cb *osEventFlagsId;
 
/// \details Mutex ID identifies the mutex (pointer to a mutex control block).
/// \note CAN BE CHANGED: \b os_mutex_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_mutex_cb *osMutexId;
 
/// \details Semaphore ID identifies the semaphore (pointer to a semaphore control block).
/// \note CAN BE CHANGED: \b os_semaphore_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_semaphore_cb *osSemaphoreId;
 
/// \details Memory Pool ID identifies the memory pool (pointer to a memory pool control block).
/// \note CAN BE CHANGED: \b os_memory_pool_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_memory_pool_cb *osMemoryPoolId;
 
/// \details Message Queue ID identifies the message queue (pointer to a message queue control block).
/// \note CAN BE CHANGED: \b os_message_queue_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_message_queue_cb *osMessageQueueId;
 
/// \details Mail Queue ID identifies the mail queue (pointer to a mail queue control block).
/// \note CAN BE CHANGED: \b os_mail_queue_cb is implementation specific in every CMSIS-RTOS.
typedef struct os_mail_queue_cb *osMailQueueId;
 
#ifdef  osCMSIS_API_V1
#define osPoolId     osMemoryPoolId
#define osMessageQId osMessageQueueId
#define osMailQId    osMailQueueId
#endif
 
 
#ifdef  osCMSIS_API_V1
 
/// Thread Definition structure contains startup information of a thread.
/// \note CAN BE CHANGED: \b os_thread_def is implementation specific in every CMSIS-RTOS.
typedef struct os_thread_def {
  os_pthread               pthread;    ///< start address of thread function
  osPriority             tpriority;    ///< initial thread priority
  uint32_t               instances;    ///< maximum number of instances of that thread function
  uint32_t               stacksize;    ///< stack size requirements in bytes; 0 is default stack size
} osThreadDef_t;
 
/// Timer Definition structure contains timer parameters.
/// \note CAN BE CHANGED: \b os_timer_def is implementation specific in every CMSIS-RTOS.
typedef struct os_timer_def {
  os_ptimer                 ptimer;    ///< start address of a timer function
} osTimerDef_t;
 
/// Mutex Definition structure contains setup information for a mutex.
/// \note CAN BE CHANGED: \b os_mutex_def is implementation specific in every CMSIS-RTOS.
typedef struct os_mutex_def {
  uint32_t                   dummy;    ///< dummy value
} osMutexDef_t;
 
/// Semaphore Definition structure contains setup information for a semaphore.
/// \note CAN BE CHANGED: \b os_semaphore_def is implementation specific in every CMSIS-RTOS.
typedef struct os_semaphore_def {
  uint32_t                   dummy;    ///< dummy value
} osSemaphoreDef_t;
 
/// Definition structure for memory block allocation.
/// \note CAN BE CHANGED: \b os_pool_def is implementation specific in every CMSIS-RTOS.
typedef struct os_pool_def {
  uint32_t                 pool_sz;    ///< number of items (elements) in the pool
  uint32_t                 item_sz;    ///< size of an item
  void                       *pool;    ///< pointer to memory for pool
} osPoolDef_t;
 
/// Definition structure for message queue.
/// \note CAN BE CHANGED: \b os_messageQ_def is implementation specific in every CMSIS-RTOS.
typedef struct os_messageQ_def {
  uint32_t                queue_sz;    ///< number of elements in the queue
  void                       *pool;    ///< memory array for messages
} osMessageQDef_t;
 
/// Definition structure for mail queue.
/// \note CAN BE CHANGED: \b os_mailQ_def is implementation specific in every CMSIS-RTOS.
typedef struct os_mailQ_def {
  uint32_t                queue_sz;    ///< number of elements in the queue
  uint32_t                 item_sz;    ///< size of an item
  void                       *pool;    ///< memory array for mail
} osMailQDef_t;
 
#endif  // osCMSIS_API_V1
 
 
/// Attributes structure for thread.
typedef struct os_thread_attr {
  const char                 *name;    ///< name of the thread
  uint32_t               attr_bits;    ///< attribute bits
  osPriority              priority;    ///< initial thread priority (default: osPriorityNormal)
  void                 *stack_addr;    ///< stack address; NULL: assigned by system
  uint32_t              stack_size;    ///< stack size in bytes; 0: default stack size
  uint32_t              reserved[3];   ///< reserved (must be 0)
} osThreadAttr_t;
 
/// Attributes structure for timer.
typedef struct os_timer_attr {
  const char                 *name;    ///< name of the timer
  uint32_t               attr_bits;    ///< attribute bits
  uint32_t                reserved;    ///< reserved (must be 0)
} osTimerAttr_t;
 
/// Attributes structure for event flags.
typedef struct os_event_flags_attr {
  const char                 *name;    ///< name of the event flags
  uint32_t               attr_bits;    ///< attribute bits
  osThreadId             thread_id;    ///< registered thread for \ref osEventWait
  uint32_t                reserved;    ///< reserved (must be 0)
} osEventFlagsAttr_t;
 
/// Attributes structure for mutex.
typedef struct os_mutex_attr {
  const char                 *name;    ///< name of the mutex
  uint32_t               attr_bits;    ///< attribute bits
  uint32_t                reserved;    ///< reserved (must be 0)
} osMutexAttr_t;
 
/// Attributes structure for semaphore.
typedef struct os_semaphore_attr {
  const char                 *name;    ///< name of the semaphore
  uint32_t               attr_bits;    ///< attribute bits
  uint32_t                reserved;    ///< reserved (must be 0)
} osSemaphoreAttr_t;
 
/// Attributes structure for memory pool.
typedef struct os_memory_pool_attr {
  const char                 *name;    ///< name of the memory pool
  uint32_t               attr_bits;    ///< attribute bits
  uint32_t                reserved;    ///< reserved (must be 0)
} osMemoryPoolAttr_t;
 
/// Attributes structure for message queue.
typedef struct os_message_queue_attr {
  const char                 *name;    ///< name of the message queue
  uint32_t               attr_bits;    ///< attribute bits
  osThreadId             thread_id;    ///< registered thread for \ref osEventWait
  uint32_t                reserved;    ///< reserved (must be 0)
} osMessageQueueAttr_t;
 
/// Attributes structure for mail queue.
typedef struct os_mail_queue_attr {
  const char                 *name;    ///< name of the mail queue
  uint32_t               attr_bits;    ///< attribute bits
  osThreadId             thread_id;    ///< registered thread for \ref osEventWait
  uint32_t                reserved;    ///< reserved (must be 0)
} osMailQueueAttr_t;
 
 
#ifdef  osCMSIS_API_V1
/// Event structure contains detailed information about an event.
typedef struct {
  osStatus                 status;     ///< status code: event or error information
  union {
    uint32_t                    v;     ///< message as 32-bit value
    void                       *p;     ///< message or mail as void pointer
    int32_t               signals;     ///< signal flags
  } value;                             ///< event value
  union {
    osMailQId             mail_id;     ///< mail id obtained by \ref osMailCreate
    osMessageQId       message_id;     ///< message id obtained by \ref osMessageCreate
  } def;                               ///< event definition
} osEvent;
#endif
 
 
//  ==== Kernel Management Functions ====
 
/// Initialize the RTOS Kernel for creating objects.
/// \return status code that indicates the execution status of the function.
osStatus osKernelInitialize (void);
 
/// Start the RTOS Kernel scheduler.
/// \return status code that indicates the execution status of the function.
osStatus osKernelStart (void);
 
/// Suspend the RTOS Kernel scheduler.
/// \param[in]     tickless      1=tickless, 0=non-tickless
/// \return time in millisec, for how long the system can sleep or power-down.
uint32_t osKernelSuspend (uint32_t tickless);
 
/// Resume the RTOS Kernel scheduler.
/// \param[in]     sleep_time    time in millisec for how long the system was in sleep or power-down mode.
/// \return status code that indicates the execution status of the function.
osStatus osKernelResume (uint32_t sleep_time);
 
/// Get the current Kernel state.
/// \return current Kernel state.
osKernelState osKernelGetState (void);
 
#ifdef  osCMSIS_API_V1
 
/// Check if the RTOS kernel is already started.
/// \return 0 RTOS is not started, 1 RTOS is started.
int32_t osKernelRunning(void);
 
#endif
 
/// Get the RTOS kernel time.
/// \return RTOS kernel current time in millisec.
uint64_t osKernelTime (void);
 
#if (defined(osFeature_KernelSysTick) && (osFeature_KernelSysTick != 0))  // Kernel System Timer available
 
/// Get the RTOS kernel system timer counter 
/// \return RTOS kernel system timer as 32-bit value 
uint32_t osKernelSysTick (void);
 
/// The RTOS kernel system timer frequency in Hz
/// \note Reflects the system timer setting and is typically defined in a configuration file.
#define osKernelSysTickFrequency 100000000
 
/// Convert a microseconds value to a RTOS kernel system timer value.
/// \param         microsec     time value in microseconds.
/// \return time value normalized to the \ref osKernelSysTickFrequency
#define osKernelSysTickMicroSec(microsec) (((uint64_t)microsec * (osKernelSysTickFrequency)) / 1000000)
 
#endif  // Kernel System Timer available
 
 
//  ==== Thread Management Functions ====
 
#ifdef  osCMSIS_API_V1
 
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
const osThreadDef_t os_thread_def_##name =               \
{ (name), (priority), (instances), (stacksz) }
#endif
 
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
osThreadId osThreadCreate (const osThreadDef_t *thread_def, void *argument);
 
#endif  // osCMSIS_API_V1
 
/// Thread attributes initialization
/// \param         name          name of the thread.
/// \param         attr_bits     attribute bits.
/// \param         priority      initial thread priority.
/// \param         stack_addr    stack address; NULL: assigned by system.
/// \param         stack_size    stack size in bytes; 0: default stack size.
#define osThreadAttrInit(name, attr_bits, priority, stack_addr, stack_size) \
  { (name), (attr_bits), (priority), (stack_addr), (stack_size), 0U, 0U, 0U }
 
/// User memory allocation for Stack 
/// \param         var           name of variable.
/// \param         stack_size    stack size in bytes.
#define osThreadStack(var, stack_size) \
  uint64_t var[((stack_size)+7)/8]
 
/// Create a thread and add it to Active Threads and set it to state READY.
/// \param[in]     pthread       thread function.
/// \param[in]     argument      pointer that is passed to the thread function as start argument.
/// \param[in]     attr          thread attributes; NULL: default values.
/// \return thread ID for reference by other functions or NULL in case of error.
osThreadId osThreadNew (os_pthread pthread, void *argument, const osThreadAttr_t *attr);
 
/// Return the thread ID of the current running thread.
/// \return thread ID for reference by other functions or NULL in case of error.
osThreadId osThreadGetId (void);
 
/// Get current thread state of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \return current thread state of the specified thread.
osThreadState osThreadGetState (osThreadId thread_id);
 
/// Change priority of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \param[in]     priority      new priority value for the thread function.
/// \return status code that indicates the execution status of the function.
osStatus osThreadSetPriority (osThreadId thread_id, osPriority priority);
 
/// Get current priority of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \return current priority value of the specified thread.
osPriority osThreadGetPriority (osThreadId thread_id);
 
/// Pass control to next thread that is in state \b READY.
/// \return status code that indicates the execution status of the function.
osStatus osThreadYield (void);
 
/// Suspend execution of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \return status code that indicates the execution status of the function.
osStatus osThreadSuspend (osThreadId thread_id);
 
/// Resume execution of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \return status code that indicates the execution status of the function.
osStatus osThreadResume (osThreadId thread_id);
 
/// Terminate execution of a thread and remove it from Active Threads.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \return status code that indicates the execution status of the function.
osStatus osThreadTerminate (osThreadId thread_id);
 
 
//  ==== Thread Flags Functions ====
 
/// Set the specified Thread Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \param[in]     flags         specifies the flags of the thread that shall be set.
/// \return status code that indicates the execution status of the function.
osStatus osThreadFlagsSet (osThreadId thread_id, int32_t flags);
 
/// Clear the specified Thread Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \param[in]     flags         specifies the flags of the thread that shall be cleared.
/// \return status code that indicates the execution status of the function.
osStatus osThreadFlagsClear (osThreadId thread_id, int32_t flags);
 
/// Get the current Thread Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadNew or \ref osThreadGetId.
/// \return current thread flags or error code if negative.
int32_t osThreadFlagsGet (osThreadId thread_id);
 
/// Wait for one or more Thread Flags of the current running thread to become signaled.
/// \param[in]     flags         specifies the flags to wait for.
/// \param[in]     options       specifies flags options (osFlagsXxxx).
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return thread flags or error code if negative.
int32_t osThreadFlagsWait (int32_t flags, uint32_t options, uint32_t millisec);
 
 
#ifdef  osCMSIS_API_V1
 
//  ==== Signal Management ====
 
/// Set the specified Signal Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     signals       specifies the signal flags of the thread that should be set.
/// \return previous signal flags of the specified thread or 0x80000000 in case of incorrect parameters.
int32_t osSignalSet (osThreadId thread_id, int32_t signals);
 
/// Clear the specified Signal Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     signals       specifies the signal flags of the thread that shall be cleared.
/// \return previous signal flags of the specified thread or 0x80000000 in case of incorrect parameters or call from ISR.
int32_t osSignalClear (osThreadId thread_id, int32_t signals);
 
/// Wait for one or more Signal Flags to become signaled for the current \b RUNNING thread.
/// \param[in]     signals       wait until all specified signal flags set or 0 for any single signal flag.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return event flag information or error code.
osEvent osSignalWait (int32_t signals, uint32_t millisec);
 
#endif  // osCMSIS_API_V1
 

//  ==== Generic Wait Functions ====
 
/// Wait for Timeout (Time Delay).
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue "time delay" value
/// \return status code that indicates the execution status of the function.
osStatus osDelay (uint32_t millisec);
 
/// Wait until specified time.
/// \param[in]     millisec      absolute time in millisec
/// \return status code that indicates the execution status of the function.
osStatus osDelayUntil (uint64_t millisec);
 
#if (defined(osFeature_EventWait) && (osFeature_EventWait != 0))  // osEventWait available
 
/// Wait for Thread Flags, Event Flags, Message, Mail, or Timeout.
/// \param[in] millisec          \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out
/// \return event bit-mask (osEventXxxx).
uint32_t osEventWait (uint32_t millisec);
 
#endif  // osEventWait available
 
 
//  ==== Timer Management Functions ====
 
#ifdef  osCMSIS_API_V1
 
/// Define a Timer object.
/// \param         name          name of the timer object.
/// \param         function      name of the timer call back function.
/// \note CAN BE CHANGED: The parameter to \b osTimerDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osTimerDef(name, function)       \
extern const osTimerDef_t os_timer_def_##name
#else                            // define the object
#define osTimerDef(name, function)       \
const osTimerDef_t os_timer_def_##name = \
{ (function) }
#endif
 
/// Access a Timer definition.
/// \param         name          name of the timer object.
/// \note CAN BE CHANGED: The parameter to \b osTimer shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osTimer(name) \
&os_timer_def_##name
 
/// Create and Initialize a timer.
/// \param[in]     timer_def     timer object referenced with \ref osTimer.
/// \param[in]     type          osTimerOnce for one-shot or osTimerPeriodic for periodic behavior.
/// \param[in]     argument      argument to the timer call back function.
/// \return timer ID for reference by other functions or NULL in case of error.
osTimerId osTimerCreate (const osTimerDef_t *timer_def, os_timer_type type, void *argument);
 
#endif  // osCMSIS_API_V1
 
/// Timer attributes initialization
/// \param         name          name of the timer.
/// \param         attr_bits     attribute bits.
#define osTimerAttrInit(name, attr_bits) \
  { (name), (attr_bits), 0U }
 
/// Create and Initialize a timer.
/// \param[in]     ptimer        start address of a timer call back function.
/// \param[in]     type          osTimerOnce for one-shot or osTimerPeriodic for periodic behavior.
/// \param[in]     argument      argument to the timer call back function.
/// \param[in]     attr          timer attributes; NULL: default values.
/// \return timer ID for reference by other functions or NULL in case of error.
osTimerId osTimerNew (os_ptimer ptimer, os_timer_type type, void *argument, const osTimerAttr_t *attr);
 
/// Start or restart a timer.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerNew.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue "time delay" value of the timer.
/// \return status code that indicates the execution status of the function.
osStatus osTimerStart (osTimerId timer_id, uint32_t millisec);
 
/// Stop a timer.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerNew.
/// \return status code that indicates the execution status of the function.
osStatus osTimerStop (osTimerId timer_id);
 
/// Check if a timer is running.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerNew.
/// \return 0 not running, 1 running, error code if negative.
int32_t osTimerIsRunning (osTimerId timer_id);
 
/// Delete a timer.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerNew.
/// \return status code that indicates the execution status of the function.
osStatus osTimerDelete (osTimerId timer_id);
 
 
//  ==== Event Flags Management Functions ====
 
#if (defined(osFeature_EventFlags) && (osFeature_EventFlags != 0U))  // Event Flags available
 
/// Event Flags attributes initialization
/// \param         name          name of the event flags.
/// \param         attr_bits     attribute bits.
/// \param         thread_id     registered thread for \ref osEventWait.
#define osEventFlagsAttrInit(name, attr_bits, thread_id) \
  { (name), (attr_bits), (thread_id), 0U }
 
/// Create and Initialize an Event Flags object.
/// \param[in]     attr          event flags attributes; NULL: default values.
/// \return event flags ID for reference by other functions or NULL in case of error.
osEventFlagsId osEventFlagsNew (const osEventFlagsAttr_t *attr);
 
/// Set the specified Event Flags.
/// \param[in]     flags_id      event flags ID obtained by \ref osEventFlagsNew.
/// \param[in]     flags         specifies the flags that shall be set.
/// \return status code that indicates the execution status of the function.
osStatus osEventFlagsSet (osEventFlagsId flags_id, int32_t flags);
 
/// Clear the specified Event Flags.
/// \param[in]     flags_id      event flags ID obtained by \ref osEventFlagsNew.
/// \param[in]     flags         specifies the flags that shall be cleared.
/// \return status code that indicates the execution status of the function.
osStatus osEventFlagsClear (osEventFlagsId flags_id, int32_t flags);
 
/// Get the current Event Flags.
/// \param[in]     flags_id      event flags ID obtained by \ref osEventFlagsNew.
/// \return current event flags or error code if negative.
int32_t osEventFlagsGet (osEventFlagsId flags_id);
 
/// Wait for one or more Event Flags to become signaled.
/// \param[in]     flags_id      event flags ID obtained by \ref osEventFlagsNew.
/// \param[in]     flags         specifies the flags to wait for.
/// \param[in]     options       specifies flags options (osFlagsXxxx).
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return event flags or error code if negative.
int32_t osEventFlagsWait (osEventFlagsId flags_id, int32_t flags, uint32_t options, uint32_t millisec);
 
/// Delete an Event Flags object.
/// \param[in]     flags_id      event flags ID obtained by \ref osEventFlagsNew.
/// \return status code that indicates the execution status of the function.
osStatus osEventFlagsDelete (osEventFlagsId flags_id);
 
#endif  // Event Flags available
 
 
//  ==== Mutex Management Functions ====
 
#ifdef  osCMSIS_API_V1
 
/// Define a Mutex.
/// \param         name          name of the mutex object.
/// \note CAN BE CHANGED: The parameter to \b osMutexDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osMutexDef(name)  \
extern const osMutexDef_t os_mutex_def_##name
#else                            // define the object
#define osMutexDef(name)  \
const osMutexDef_t os_mutex_def_##name = { 0 }
#endif
 
/// Access a Mutex definition.
/// \param         name          name of the mutex object.
/// \note CAN BE CHANGED: The parameter to \b osMutex shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMutex(name)  \
&os_mutex_def_##name
 
/// Create and Initialize a Mutex object.
/// \param[in]     mutex_def     mutex definition referenced with \ref osMutex.
/// \return mutex ID for reference by other functions or NULL in case of error.
osMutexId osMutexCreate (const osMutexDef_t *mutex_def);
 
#define osMutexWait osMutexAcquire

#endif  // osCMSIS_API_V1
 
/// Mutex attributes initialization
/// \param         name          name of the mutex.
/// \param         attr_bits     attribute bits.
#define osMutexAttrInit(name, attr_bits) \
  { (name), (attr_bits), 0U }
 
/// Create and Initialize a Mutex object.
/// \param[in]     attr          mutex attributes; NULL: default values.
/// \return mutex ID for reference by other functions or NULL in case of error.
osMutexId osMutexNew (const osMutexAttr_t *attr);
 
/// Acquire a Mutex or timeout if it is locked.
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexNew.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
osStatus osMutexAcquire (osMutexId mutex_id, uint32_t millisec);
 
/// Release a Mutex that was acquired by \ref osMutexAcquire.
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexNew.
/// \return status code that indicates the execution status of the function.
osStatus osMutexRelease (osMutexId mutex_id);
 
/// Delete a Mutex object.
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexNew.
/// \return status code that indicates the execution status of the function.
osStatus osMutexDelete (osMutexId mutex_id);
 
 
//  ==== Semaphore Management Functions ====
 
#ifdef  osCMSIS_API_V1
 
/// Define a Semaphore object.
/// \param         name          name of the semaphore object.
/// \note CAN BE CHANGED: The parameter to \b osSemaphoreDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osSemaphoreDef(name)  \
extern const osSemaphoreDef_t os_semaphore_def_##name
#else                            // define the object
#define osSemaphoreDef(name)  \
const osSemaphoreDef_t os_semaphore_def_##name = { 0 }
#endif
 
/// Access a Semaphore definition.
/// \param         name          name of the semaphore object.
/// \note CAN BE CHANGED: The parameter to \b osSemaphore shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osSemaphore(name)  \
&os_semaphore_def_##name
 
/// Create and Initialize a Semaphore object.
/// \param[in]     semaphore_def semaphore definition referenced with \ref osSemaphore.
/// \param[in]     count         maximum and initial number of available tokens.
/// \return semaphore ID for reference by other functions or NULL in case of error.
osSemaphoreId osSemaphoreCreate (const osSemaphoreDef_t *semaphore_def, int32_t count);
 
/// Wait until a Semaphore token becomes available.
/// \param[in]     semaphore_id  semaphore object referenced with \ref osSemaphoreCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return number of available tokens, or -1 in case of incorrect parameters.
int32_t osSemaphoreWait (osSemaphoreId semaphore_id, uint32_t millisec);
 
#endif  // osCMSIS_API_V1
 
/// Semaphore attributes initialization
/// \param         name          name of the semaphore.
/// \param         attr_bits     attribute bits.
#define osSemaphoreAttrInit(name, attr_bits) \
  { (name), (attr_bits), 0U }
 
/// Create and Initialize a Semaphore object.
/// \param[in]     max_count     maximum number of available tokens.
/// \param[in]     initial_count initial number of available tokens.
/// \param[in]     attr          semaphore attributes; NULL: default values.
/// \return semaphore ID for reference by other functions or NULL in case of error.
osSemaphoreId osSemaphoreNew (uint32_t max_count, uint32_t initial_count, const osSemaphoreAttr_t *attr);
 
/// Acquire a Semaphore token or timeout if no tokens are available.
/// \param[in]     semaphore_id  semaphore ID obtained by \ref osSemaphoreNew.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
osStatus osSemaphoreAcquire (osSemaphoreId semaphore_id, uint32_t millisec);
 
/// Release a Semaphore token that was acquired by \ref osSemaphoreAcquire.
/// \param[in]     semaphore_id  semaphore ID obtained by \ref osSemaphoreNew.
/// \return status code that indicates the execution status of the function.
osStatus osSemaphoreRelease (osSemaphoreId semaphore_id);
 
/// Delete a Semaphore object.
/// \param[in]     semaphore_id  semaphore ID obtained by \ref osSemaphoreNew.
/// \return status code that indicates the execution status of the function.
osStatus osSemaphoreDelete (osSemaphoreId semaphore_id);
 
 
//  ==== Memory Pool Management Functions ====
 
#if (defined(osFeature_MemoryPool) && (osFeature_MemoryPool != 0))  // Memory Pool available
 
#ifdef  osCMSIS_API_V1
 
/// \brief Define a Memory Pool.
/// \param         name          name of the memory pool.
/// \param         no            maximum number of blocks (objects) in the memory pool.
/// \param         type          data type of a single block (object).
/// \note CAN BE CHANGED: The parameter to \b osPoolDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osPoolDef(name, no, type)      \
extern const osPoolDef_t os_pool_def_##name
#else                            // define the object
#define osPoolDef(name, no, type)      \
const osPoolDef_t os_pool_def_##name = \
{ (no), sizeof(type), NULL }
#endif
 
/// \brief Access a Memory Pool definition.
/// \param         name          name of the memory pool
/// \note CAN BE CHANGED: The parameter to \b osPool shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osPool(name) \
&os_pool_def_##name
 
/// Create and Initialize a Memory Pool object.
/// \param[in]     pool_def      memory pool definition referenced with \ref osPool.
/// \return memory pool ID for reference by other functions or NULL in case of error.
osPoolId osPoolCreate (const osPoolDef_t *pool_def);
 
/// Allocate a memory block from a Memory Pool and set memory block to zero.
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \return address of the allocated memory block or NULL in case of no memory available.
void *osPoolCAlloc (osPoolId pool_id);
 
#define osPoolAlloc osMemoryPoolAlloc
#define osPoolFree  osMemoryPoolFree
 
#endif  // osCMSIS_API_V1
 
/// Memory Pool attributes initialization
/// \param         name          name of the memory pool.
/// \param         attr_bits     attribute bits.
#define osMemoryPoolAttrInit(name, attr_bits) \
  { (name), (attr_bits), 0U }
 
/// User memory allocation for Memory Pool data
/// \param         var           name of variable.
/// \param         block_max     maximum number of memory blocks in memory pool.
/// \param         block_size    size of a memory block in bytes.
#define osMemoryPoolMem(var, block_max, block_size) \
  uint32_t var[(block_max)*(((block_size)+3)/4)]
 
/// Create and Initialize a Memory Pool object.
/// \param[in]     block_max     maximum number of memory blocks in memory pool.
/// \param[in]     block_size    size of a memory block in bytes.
/// \param[in]     memory        pointer to pool memory.
/// \param[in]     attr          memory pool attributes; NULL: default values.
/// \return memory pool ID for reference by other functions or NULL in case of error.
osMemoryPoolId osMemoryPoolNew (uint32_t block_max, uint32_t block_size, void *memory, const osMemoryPoolAttr_t *attr);
 
/// Allocate a memory block from a Memory Pool.
/// \param[in]     pool_id       memory pool ID obtained by \ref osMemoryPoolNew.
/// \return address of the allocated memory block or NULL in case of no memory is available.
void *osMemoryPoolAlloc (osMemoryPoolId pool_id);
 
/// Return an allocated memory block back to a Memory Pool.
/// \param[in]     pool_id       memory pool ID obtained by \ref osMemoryPoolNew.
/// \param[in]     block         address of the allocated memory block to be returned to the memory pool.
/// \return status code that indicates the execution status of the function.
osStatus osMemoryPoolFree (osMemoryPoolId pool_id, void *block);
 
/// Get a Memory Pool information.
/// \param[in]     pool_id       memory pool ID obtained by \ref osMemoryPoolNew.
/// \param[out]    block_max     pointer to buffer for maximum number of memory blocks in memory pool.
/// \param[out]    block_size    pointer to buffer for size of a memory block in bytes.
/// \param[out]    block_used    pointer to buffer for number of used memory blocks.
/// \return status code that indicates the execution status of the function.
osStatus osMemoryPoolGetInfo (osMemoryPoolId pool_id, uint32_t *block_max, uint32_t *block_size, uint32_t *block_used);
 
/// Delete a Memory Pool object.
/// \param[in]     pool_id       memory pool ID obtained by \ref osMemoryPoolNew.
/// \return status code that indicates the execution status of the function.
osStatus osMemoryPoolDelete (osMemoryPoolId pool_id);
 
#endif  // Memory Pool available
 
 
//  ==== Message Queue Management Functions ====
 
#if (defined(osFeature_MessageQueue) && (osFeature_MessageQueue != 0))  // Message Queue available
 
#ifdef  osCMSIS_API_V1
 
/// \brief Create a Message Queue Definition.
/// \param         name          name of the queue.
/// \param         queue_sz      maximum number of messages in the queue.
/// \param         type          data type of a single message element (for debugger).
/// \note CAN BE CHANGED: The parameter to \b osMessageQDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osMessageQDef(name, queue_sz, type)    \
extern const osMessageQDef_t os_messageQ_def_##name
#else                            // define the object
#define osMessageQDef(name, queue_sz, type)    \
const osMessageQDef_t os_messageQ_def_##name = \
{ (queue_sz), NULL }
#endif
 
/// \brief Access a Message Queue Definition.
/// \param         name          name of the queue
/// \note CAN BE CHANGED: The parameter to \b osMessageQ shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMessageQ(name) \
&os_messageQ_def_##name
 
/// Create and Initialize a Message Queue object.
/// \param[in]     queue_def     message queue definition referenced with \ref osMessageQ.
/// \param[in]     thread_id     thread ID (obtained by \ref osThreadCreate or \ref osThreadGetId) or NULL.
/// \return message queue ID for reference by other functions or NULL in case of error.
osMessageQId osMessageCreate (const osMessageQDef_t *queue_def, osThreadId thread_id);
 
/// Get a Message from a Queue or timeout if Queue is empty.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return event information that includes status code.
osEvent osMessageGet (osMessageQId queue_id, uint32_t millisec);
 
#define osMessagePut osMessageQueuePut
 
#endif  // osCMSIS_API_V1
 
/// Message Queue attributes initialization
/// \param         name          name of the message queue.
/// \param         attr_bits     attribute bits.
/// \param         thread_id     registered thread for \ref osEventWait.
#define osMessageQueueAttrInit(name, attr_bits, thread_id) \
  { (name), (attr_bits), (thread_id), 0U }
 
/// User memory allocation for Message Queue data
/// \param         var           name of variable.
/// \param         queue_size    maximum number of messages in queue.
#define osMessageQueueMem(var, queue_size) \
  uint32_t var[(queue_size)]
 
/// Create and Initialize a Message Queue object.
/// \param[in]     queue_size    maximum number of messages in queue.
/// \param[in]     memory        pointer to message memory pool.
/// \param[in]     attr          message queue attributes; NULL: default values.
/// \return message queue ID for reference by other functions or NULL in case of error.
osMessageQueueId osMessageQueueNew (uint32_t queue_size, void *memory, const osMessageQueueAttr_t *attr);
 
/// Put a Message into a Queue or timeout if Queue is full.
/// \param[in]     queue_id      message queue ID obtained by \ref osMessageQueueNew.
/// \param[in]     message       message (32-bit value) to put into a queue.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
osStatus osMessageQueuePut (osMessageQueueId queue_id, uint32_t message, uint32_t millisec);
 
/// Get a Message from a Queue or timeout if Queue is empty.
/// \param[in]     queue_id      message queue ID obtained by \ref osMessageQueueNew.
/// \param[out]    message       pointer to buffer for message to get from a queue.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
osStatus osMessageQueueGet (osMessageQueueId queue_id, uint32_t *message, uint32_t millisec);
 
/// Get a Message Queue information.
/// \param[in]     queue_id      message queue ID obtained by \ref osMessageQueueNew.
/// \param[out]    queue_size    pointer to buffer for maximum number of messages in a queue.
/// \param[out]    message_count pointer to buffer for number of messages in a queue.
/// \return status code that indicates the execution status of the function.
osStatus osMessageQueueGetInfo (osMessageQueueId queue_id, uint32_t *queue_size, uint32_t *message_count);
 
/// Reset a Message Queue to initial empty state.
/// \param[in]     queue_id      message queue ID obtained by \ref osMessageQueueNew.
/// \return status code that indicates the execution status of the function.
osStatus osMessageQueueReset (osMessageQueueId queue_id);
 
/// Delete a Message Queue object.
/// \param[in]     queue_id      message queue ID obtained by \ref osMessageQueueNew.
/// \return status code that indicates the execution status of the function.
osStatus osMessageQueueDelete (osMessageQueueId queue_id);
 
#endif  // Message Queue available
 
 
//  ==== Mail Queue Management Functions ====
 
#if (defined(osFeature_MailQueue) && (osFeature_MailQueue != 0))  // Mail Queue available
 
#ifdef  osCMSIS_API_V1
 
/// \brief Create a Mail Queue Definition.
/// \param         name          name of the queue.
/// \param         queue_sz      maximum number of mails in the queue.
/// \param         type          data type of a single mail element.
/// \note CAN BE CHANGED: The parameter to \b osMailQDef shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#if defined (osObjectsExternal)  // object is external
#define osMailQDef(name, queue_sz, type) \
extern const osMailQDef_t os_mailQ_def_##name
#else                            // define the object
#define osMailQDef(name, queue_sz, type) \
const osMailQDef_t os_mailQ_def_##name = \
{ (queue_sz), sizeof(type), NULL }
#endif
 
/// \brief Access a Mail Queue Definition.
/// \param         name          name of the queue
/// \note CAN BE CHANGED: The parameter to \b osMailQ shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osMailQ(name) \
&os_mailQ_def_##name
 
/// Create and Initialize a Mail Queue object.
/// \param[in]     queue_def     mail queue definition referenced with \ref osMailQ.
/// \param[in]     thread_id     thread ID (obtained by \ref osThreadCreate or \ref osThreadGetId) or NULL.
/// \return mail queue ID for reference by other functions or NULL in case of error.
osMailQId osMailCreate (const osMailQDef_t *queue_def, osThreadId thread_id);
 
/// Allocate a memory block for mail from a mail memory pool and set memory block to zero.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out
/// \return pointer to memory block that can be filled with mail or NULL in case of error.
void *osMailCAlloc (osMailQId queue_id, uint32_t millisec);
 
/// Get a Mail from a Queue or timeout if Queue is empty.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return event information that includes status code.
osEvent osMailGet (osMailQId queue_id, uint32_t millisec);
 
#define osMailAlloc osMailQueueAlloc
#define osMailPut   osMailQueuePut
#define osMailFree  osMailQueueFree
 
#endif  // osCMSIS_API_V1
 
/// Mail Queue attributes initialization
/// \param         name          name of the mail queue.
/// \param         attr_bits     attribute bits.
/// \param         thread_id     registered thread for \ref osEventWait.
#define osMailQueueAttrInit(name, attr_bits, thread_id) \
  { (name), (attr_bits), (thread_id), 0U }
 
/// User memory allocation for Mail Queue data
/// \param         var           name of variable.
/// \param         queue_size    maximum number of mails in queue.
/// \param         mail_size     size of a mail in bytes.
#define osMailQueueMem(var, queue_size, mail_size) \
  uint32_t var[(queue_size)+((queue_size)*(((mail_size)+3)/4))]
 
/// Create and Initialize a Mail Queue object.
/// \param[in]     queue_size    maximum number of mails in queue.
/// \param[in]     mail_size     size of a mail in bytes.
/// \param[in]     memory        pointer to mail memory pool.
/// \param[in]     attr          mail queue attributes; NULL: default values.
/// \return mail queue ID for reference by other functions or NULL in case of error.
osMailQueueId osMailQueueNew (uint32_t queue_size, uint32_t mail_size, void *memory, const osMailQueueAttr_t *attr);
 
/// Allocate a memory block for mail from a mail memory pool.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailQueueNew.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out
/// \return pointer to memory block that can be filled with mail or NULL in case of error.
void *osMailQueueAlloc (osMailQueueId queue_id, uint32_t millisec);
 
/// Put a Mail into a Queue.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailQueueNew.
/// \param[in]     mail          pointer to memory with mail to put into a queue.
/// \return status code that indicates the execution status of the function.
osStatus osMailQueuePut (osMailQueueId queue_id, const void *mail);
 
/// Get a Mail from a Queue or timeout if Queue is empty.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailQueueNew.
/// \param[out]    mail          pointer to buffer for mail to get from a queue.
/// \param[in]     millisec      \ref CMSIS_RTOS_TimeOutValue or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
osStatus osMailQueueGet (osMailQueueId queue_id, void *mail, uint32_t millisec);
 
/// Free a memory block by returning it to a mail memory pool.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailQueueNew.
/// \param[in]     mail          pointer to memory block allocated with \ref osMailQueueAlloc.
/// \return status code that indicates the execution status of the function.
osStatus osMailQueueFree (osMailQueueId queue_id, void *mail);
 
/// Get a Mail Queue information.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailQueueNew.
/// \param[out]    queue_size    pointer to buffer for maximum number of mails in a queue.
/// \param[out]    mail_size     pointer to buffer for size of a mail in bytes.
/// \param[out]    mail_count    pointer to buffer for number of mails in a queue.
/// \return status code that indicates the execution status of the function.
osStatus osMailQueueGetInfo (osMailQueueId queue_id, uint32_t *queue_size, uint32_t *mail_size, uint32_t *mail_count);
 
/// Reset a Mail Queue to initial empty state.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailQueueNew.
/// \return status code that indicates the execution status of the function.
osStatus osMailQueueReset (osMailQueueId queue_id);
 
/// Delete a Mail Queue object.
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailQueueNew.
/// \return status code that indicates the execution status of the function.
osStatus osMailQueueDelete (osMailQueueId queue_id);
 
#endif  // Mail Queue available
 
 
#ifdef  __cplusplus
}
#endif
 
#endif  // _CMSIS_OS_H
