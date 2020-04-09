#include "Timing.h"

#ifdef CORTEXM

#define SYSTICK_INITIAL_VALUE       0xFFFFFF
static uint32_t startCycles=0;

#if   defined ARMCM0
  #include "ARMCM0.h"
#elif defined ARMCM0P
  #include "ARMCM0plus.h"
#elif defined ARMCM0P_MPU
  #include "ARMCM0plus_MPU.h"
#elif defined ARMCM3
  #include "ARMCM3.h"
#elif defined ARMCM4
  #include "ARMCM4.h"
#elif defined ARMCM4_FP
  #include "ARMCM4_FP.h"
#elif defined ARMCM7
  #include "ARMCM7.h" 
#elif defined ARMCM7_SP
  #include "ARMCM7_SP.h"
#elif defined ARMCM7_DP
  #include "ARMCM7_DP.h"
#elif defined (ARMCM33_DSP_FP)
  #include "ARMCM33_DSP_FP.h"
#elif defined (ARMCM33_DSP_FP_TZ)
  #include "ARMCM33_DSP_FP_TZ.h"
#elif defined ARMSC000
  #include "ARMSC000.h"
#elif defined ARMSC300
  #include "ARMSC300.h"
#elif defined ARMv8MBL
  #include "ARMv8MBL.h"
#elif defined ARMv8MML
  #include "ARMv8MML.h"
#elif defined ARMv8MML_DSP
  #include "ARMv8MML_DSP.h"
#elif defined ARMv8MML_SP
  #include "ARMv8MML_SP.h"
#elif defined ARMv8MML_DSP_SP
  #include "ARMv8MML_DSP_SP.h"
#elif defined ARMv8MML_DP
  #include "ARMv8MML_DP.h"
#elif defined ARMv8MML_DSP_DP
  #include "ARMv8MML_DSP_DP.h"
#elif defined ARMv81MML_DSP_DP_MVE_FP
  #include "ARMv81MML_DSP_DP_MVE_FP.h"
#elif defined ARMv7A
  /* TODO */
#else
  #warning "no appropriate header file found!"
#endif
#endif

#ifdef CORTEXA
#include "cmsis_cp15.h"
unsigned int startCycles;

#define DO_RESET 1
#define ENABLE_DIVIDER 0 
#endif

#ifdef EXTBENCH
unsigned long sectionCounter=0;
#endif 

void initCycleMeasurement()
{
#ifdef CORTEXM
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk;
    SysTick->LOAD = SYSTICK_INITIAL_VALUE;
#endif 

#ifdef CORTEXA

    // in general enable all counters (including cycle counter)
    int32_t   value = 1;

    // peform reset:
    if (DO_RESET)
    {
        value |= 2;             // reset all counters to zero.
        value |= 4;             // reset cycle counter to zero.
    }

    if (ENABLE_DIVIDER)
        value |= 8;             // enable "by 64" divider for CCNT.

    value |= 16;

    // program the performance-counter control-register:
    __set_CP(15, 0, value, 9, 12, 0);

    // enable all counters:
    __set_CP(15, 0, 0x8000000f, 9, 12, 1);

    // clear overflows:
    __set_CP(15, 0, 0x8000000f, 9, 12, 3);
#endif

}

void cycleMeasurementStart()
{
#ifndef EXTBENCH
#ifdef CORTEXM
    /* 
    TODO:
    This code is likely to be wrong. Don't rely on it for benchmarks.

    */
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk;
    SysTick->LOAD = SYSTICK_INITIAL_VALUE;

    SysTick->CTRL = SysTick_CTRL_ENABLE_Msk | SysTick_CTRL_CLKSOURCE_Msk;  

    while(SysTick->VAL == 0);

    startCycles = SysTick->VAL;


    
#endif

#ifdef CORTEXA
    unsigned int value;
    // Read CCNT Register
    __get_CP(15, 0, value, 9, 13, 0);
    startCycles =  value;
#endif
#endif 

}

void cycleMeasurementStop()
{
#ifndef EXTBENCH
#ifdef CORTEXM
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk;
    SysTick->LOAD = SYSTICK_INITIAL_VALUE;
#endif
#endif
}

Testing::cycles_t getCycles()
{
#ifdef CORTEXM
    uint32_t v = SysTick->VAL;
    return(startCycles - v);
#endif 

#ifdef CORTEXA
    unsigned int value;
    // Read CCNT Register
    __get_CP(15, 0, value, 9, 13, 0);
    return(value - startCycles);
#endif

}
