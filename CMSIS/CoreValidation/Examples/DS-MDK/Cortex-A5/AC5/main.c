/*----------------------------------------------------------------------------
 * Name:    main.c
 *----------------------------------------------------------------------------*/

/* Includes ------------------------------------------------------------------*/

#include <stdio.h>

#include "RTE_Components.h"
#include  CMSIS_device_header
 
#ifdef RTE_Compiler_EventRecorder
#include "EventRecorder.h"
#endif

#include "cmsis_cv.h"


/* Private functions ---------------------------------------------------------*/
int main (void);

/**
  * @brief  Main program
  * @param  None
  * @retval None
  */
int main (void) {

  // System Initialization
  SystemCoreClockUpdate();
#ifdef RTE_Compiler_EventRecorder
  // Initialize and start Event Recorder
  (void)EventRecorderInitialize(EventRecordError, 1U);
  (void)EventRecorderEnable    (EventRecordAll, 0xFEU, 0xFEU);
#endif
  
  cmsis_cv();
  
  for(;;) {}
}
