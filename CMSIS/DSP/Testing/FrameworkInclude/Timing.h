#ifndef _TIMING_H_
#define _TIMING_H_

#include "Test.h"
#include "arm_math_types.h"
#include "arm_math_types_f16.h"

void initCycleMeasurement();
void cycleMeasurementStart();
void cycleMeasurementStop();

Testing::cycles_t getCycles();

#if defined(EXTBENCH)  || defined(CACHEANALYSIS)
extern unsigned long sectionCounter;

#if   defined ( __CC_ARM )
    #define dbgInst(imm) __asm volatile{ DBG (imm) }
#elif defined ( __GNUC__ ) || defined ( __llvm__ )
    #define dbgInst(imm) __asm volatile("DBG %0\n\t" : :"Ir" ((imm)) )
#else
    #error "Unsupported compiler"
#endif
#define startSectionNB(num) dbgInst(((num) & 0x7) | 0x8)
#define stopSectionNB(num)  dbgInst(((num) & 0x7) | 0x0)

static inline void startSection() {
    switch(sectionCounter & 0x7)
    {
      case 0:
        startSectionNB(0);
      break;
      case 1:
        startSectionNB(1);
      break;
      case 2:
        startSectionNB(2);
      break;
      case 3:
        startSectionNB(3);
      break;
      case 4:
        startSectionNB(4);
      break;
      case 5:
        startSectionNB(5);
      break;
      case 6:
        startSectionNB(6);
      break;
      case 7:
        startSectionNB(7);
      break;
      default:
        startSectionNB(0);
    }
}

static inline void stopSection() {
    switch(sectionCounter & 0x7)
     {
      case 0:
        stopSectionNB(0);
      break;
      case 1:
        stopSectionNB(1);
      break;
      case 2:
        stopSectionNB(2);
      break;
      case 3:
        stopSectionNB(3);
      break;
      case 4:
        stopSectionNB(4);
      break;
      case 5:
        stopSectionNB(5);
      break;
      case 6:
        stopSectionNB(6);
      break;
      case 7:
        stopSectionNB(7);
      break;
      default:
        stopSectionNB(0);
     }
     
     sectionCounter++;
}

#endif 

#endif
