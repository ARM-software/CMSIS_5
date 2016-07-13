
#include "cmsis_os2.h"                                           // CMSIS RTOS header file

/*----------------------------------------------------------------------------
 *      Event Flags creation & usage
 *---------------------------------------------------------------------------*/
 
void *Thread_EventSender   (void *argument);                   // thread function 1
void *Thread_EventReceiver (void *argument);                   // thread function 2
osThreadId_t tid_Thread_EventSender;                                // thread id 1
osThreadId_t tid_Thread_EventReceiver;                                // thread id 2

#define MSGQUEUE_OBJECTS      16                                // number of Message Queue Objects

typedef struct {                                                // object data type
  uint8_t Buf[32];
  uint8_t Idx;
} MEM_BLOCK_t;

typedef struct {                                                // object data type
  uint8_t Buf[32];
  uint8_t Idx;
} MSGQUEUE_OBJ_t;

  
osEventFlagsId_t evt_id;                                      // message queue id

#define FLAGS_MSK1 0x00000001ul

int Init_Events (void) {
 
  tid_Thread_EventSender = osThreadNew (Thread_EventSender, NULL, NULL);
  if (!tid_Thread_EventSender) return(-1);
  tid_Thread_EventReceiver = osThreadNew (Thread_EventReceiver, NULL, NULL);
  if (!tid_Thread_EventReceiver) return(-1);
  
  return(0);
}

void *Thread_EventSender (void *argument) {
  
  while (1) {
		evt_id = osEventFlagsNew(NULL);
		osEventFlagsSet(evt_id, FLAGS_MSK1);
    osThreadYield ();                                           // suspend thread
  }
}

void *Thread_EventReceiver (void *argument) {
  uint32_t  flags;

  while (1) {
    flags = osEventFlagsWait (evt_id,FLAGS_MSK1,osFlagsWaitAny, osWaitForever);
		  //handle event
   
  }
}
