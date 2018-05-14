/*----------------------------------------------------------------------------
 * Name:    main.c
 *----------------------------------------------------------------------------*/

/* Includes ------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include "RTE_Components.h"
#include  CMSIS_device_header
 
#ifdef RTE_Compiler_EventRecorder
#include "EventRecorder.h"
#endif

#include "cmsis_cv.h"

//lint -e970 allow using int for main

/* Private functions ---------------------------------------------------------*/
int main (void);

/**
  * @brief  Main program
  * @param  None
  * @retval None
  */
int main (void)
{  

  // System Initialization
  SystemCoreClockUpdate();
#ifdef RTE_Compiler_EventRecorder
  // Initialize and start Event Recorder
  (void)EventRecorderInitialize(EventRecordError, 1U);
  (void)EventRecorderEnable    (EventRecordAll, 0xFEU, 0xFEU);
#endif
  
  cmsis_cv();
  
  #ifdef __MICROLIB
  for(;;) {}
  #else
  exit(0);
  #endif
}

#if defined(__CORTEX_A)
#include "irq_ctrl.h"
#if (defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)) || \
  (defined ( __GNUC__ ))
__attribute__((interrupt("IRQ")))
#elif defined ( __CC_ARM )
__irq
#elif defined ( __ICCARM__ )
__irq __arm
#else
#error "Unsupported compiler!"
#endif
void IRQ_Handler(void) {
  const IRQn_ID_t irqn = IRQ_GetActiveIRQ();
  IRQHandler_t const handler = IRQ_GetHandler(irqn);
  if (handler != NULL) {
    __enable_irq();
    handler();
    __disable_irq();
  }
  IRQ_EndOfInterrupt(irqn);
}
#endif

#if defined(__CORTEX_M)
__NO_RETURN
extern void HardFault_Handler(void);
void HardFault_Handler(void) {
  #ifdef __MICROLIB
  for(;;) {}
  #else
  exit(0);
  #endif
}
#endif
