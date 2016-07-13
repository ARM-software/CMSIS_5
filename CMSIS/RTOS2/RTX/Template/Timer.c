
#include "cmsis_os2.h"                                           // CMSIS RTOS header file

/*----------------------------------------------------------------------------
 *      Timer: Sample timer functions
 *---------------------------------------------------------------------------*/
 

/*----- One-Shoot Timer Example -----*/
static void Timer1_Callback (void const *arg);                  // prototype for timer callback function

static osTimerId_t id1;                                           // timer id
static uint32_t  exec1;                                         // argument for the timer call back function

// One-Shoot Timer Function
static void Timer1_Callback (void const *arg) {
  // add user code here
}


/*----- Periodic Timer Example -----*/
static void Timer2_Callback (void const *arg);                  // prototype for timer callback function

static osTimerId_t id2;                                           // timer id
static uint32_t  exec2;                                         // argument for the timer call back function

// Periodic Timer Example
static void Timer2_Callback (void const *arg) {
  // add user code here
}


// Example: Create and Start timers
void Init_Timers (void) {
  osStatus_t status;                                              // function return status
 
  // Create one-shoot timer
  exec1 = 1;
  id1 = osTimerNew ((os_timer_func_t)&Timer1_Callback, osTimerOnce, &exec1, NULL);
  if (id1 != NULL) {    // One-shot timer created
    // start timer with delay 100ms
    status = osTimerStart (id1, 100);            
    if (status != osOK) {
      // Timer could not be started
    }
  }
 
  // Create periodic timer
  exec2 = 2;
  id2 = osTimerNew((os_timer_func_t)&Timer2_Callback, osTimerPeriodic, &exec2, NULL);
  if (id2 != NULL) {    // Periodic timer created
    // start timer with periodic 1000ms interval
    status = osTimerStart (id2, 1000);            
    if (status != osOK) {
      // Timer could not be started
    }
  }
}
