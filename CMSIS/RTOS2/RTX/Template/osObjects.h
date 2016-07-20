/*----------------------------------------------------------------------------
 * osObjects.h: CMSIS-RTOS global object definitions for an application
 *----------------------------------------------------------------------------
 *
 * This header file defines global RTOS objects used throughout a project
 *
 * #define osObjectsPublic indicates that objects are defined; without that
 * definition the objects are defined as external symbols.
 *
 *--------------------------------------------------------------------------*/


#ifndef __osObjects
#define __osObjects

#if (!defined (osObjectsPublic))
#define osObjectsExternal           // define RTOS objects with extern attribute
#endif

#include "cmsis_os2.h"               // CMSIS RTOS header file


// global 'thread' functions ---------------------------------------------------
/* 
Example:
extern void *sample_name (void *argument);         // thread function

osThreadId_t tid_sample_name;                             // thread id
*/


// global 'semaphores' ----------------------------------------------------------
/* 
Example:
osSemaphoreId_t sid_sample_name;                          // semaphore id
*/


// global 'memory pools' --------------------------------------------------------
/* 
Example:
typedef struct sample_name type_sample_name;            // object data type

osMemoryPoolId_t mpid_sample_name;                              // memory pool id
*/


// global 'message queues' -------------------------------------------------------
/* 
Example:
typedef struct sample_name type_sample_name;            // object data type

osMessageQueueId_t mid_sample_name;                           // message queue id
*/

