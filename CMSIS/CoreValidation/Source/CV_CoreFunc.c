/*-----------------------------------------------------------------------------
 *      Name:         CV_CoreFunc.c 
 *      Purpose:      CMSIS CORE validation tests implementation
 *-----------------------------------------------------------------------------
 *      Copyright (c) 2017 ARM Limited. All rights reserved.
 *----------------------------------------------------------------------------*/

#include "CV_Framework.h"
#include "cmsis_cv.h"

/*-----------------------------------------------------------------------------
 *      Test implementation
 *----------------------------------------------------------------------------*/

static volatile uint32_t irqTaken = 0U;

static void TC_CoreFunc_EnDisIRQIRQHandler(void) {
  ++irqTaken;
}

static volatile uint32_t irqIPSR = 0U;
static volatile uint32_t irqXPSR = 0U;

static void TC_CoreFunc_IPSR_IRQHandler(void) {
  irqIPSR = __get_IPSR();
  irqXPSR = __get_xPSR();
}

/*-----------------------------------------------------------------------------
 *      Test cases
 *----------------------------------------------------------------------------*/

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreFunc_EnDisIRQ
\details
- Check if __disable_irq() and __enable_irq() have expected behaviour.
*/
void TC_CoreFunc_EnDisIRQ (void)
{
  __disable_irq();
 
  NVIC_EnableIRQ(WDT_IRQn);
  NVIC_ClearPendingIRQ(WDT_IRQn);

  TST_IRQHandler = TC_CoreFunc_EnDisIRQIRQHandler;
  irqTaken = 0U;

  NVIC_SetPendingIRQ(WDT_IRQn);  
  for(uint32_t i = 10U; i > 0U; --i) {}
    
  // Interrupt is not taken
  ASSERT_TRUE(irqTaken == 0U);
  
  __enable_irq();
    
  for(uint32_t i = 10U; i > 0U; --i) {}

  // Interrupt was taken    
  ASSERT_TRUE(irqTaken == 1U);
      
  __disable_irq();
  NVIC_DisableIRQ(WDT_IRQn);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreFunc_GetCtrl
\details
- Check if __set_CONTROL and __get_CONTROL() sets/gets control register
*/
void TC_CoreFunc_Control (void) {
  // don't use stack for this variables
  static uint32_t orig;
  static uint32_t ctrl;
  static uint32_t result;

  orig = __get_CONTROL();
  ctrl = orig;
  result = UINT32_MAX;
  
#ifdef CONTROL_SPSEL_Msk   
  // toggle SPSEL
  ctrl = (ctrl & ~CONTROL_SPSEL_Msk) | (~ctrl & CONTROL_SPSEL_Msk);
#endif
  
  __set_CONTROL(ctrl);
  __ISB();

  result = __get_CONTROL();
  
  __set_CONTROL(orig);
  __ISB();
  
  ASSERT_TRUE(result == ctrl);
  ASSERT_TRUE(__get_CONTROL() == orig);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreFunc_IPSR
\details
- Check if __get_IPSR instrinsic is available
- Check if __get_xPSR instrinsic is available
- Result differentiates between thread and exception modes
*/
void TC_CoreFunc_IPSR (void) {
  uint32_t result = __get_IPSR();
  ASSERT_TRUE(result == 0U); // Thread Mode
  
  result = __get_xPSR();
  ASSERT_TRUE((result & xPSR_ISR_Msk) == 0U); // Thread Mode
  
  TST_IRQHandler = TC_CoreFunc_IPSR_IRQHandler;
  irqIPSR = 0U;
  irqXPSR = 0U;

  NVIC_ClearPendingIRQ(WDT_IRQn);
  NVIC_EnableIRQ(WDT_IRQn);
  __enable_irq();
  
  NVIC_SetPendingIRQ(WDT_IRQn);  
  for(uint32_t i = 10U; i > 0U; --i) {}

  __disable_irq();
  NVIC_DisableIRQ(WDT_IRQn);
    
  ASSERT_TRUE(irqIPSR != 0U); // Exception Mode
  ASSERT_TRUE((irqXPSR & xPSR_ISR_Msk) != 0U); // Exception Mode
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/

#if defined(__CC_ARM)
#define SUBS(Rd, Rm, Rn) __ASM("SUBS " # Rd ", " # Rm ", " # Rn)
#define ADDS(Rd, Rm, Rn) __ASM("ADDS " # Rd ", " # Rm ", " # Rn)
#elif defined( __GNUC__ )  && (defined(__ARM_ARCH_6M__) || defined(__ARM_ARCH_8M_BASE__))
#define SUBS(Rd, Rm, Rn) __ASM("SUB %0, %1, %2" : "=r"(Rd) : "r"(Rm), "r"(Rn) : "cc")
#define ADDS(Rd, Rm, Rn) __ASM("ADD %0, %1, %2" : "=r"(Rd) : "r"(Rm), "r"(Rn) : "cc")
#elif defined(_lint)
//lint -save -e(9026) allow function-like macro
#define SUBS(Rd, Rm, Rn) ((Rd) = (Rm) - (Rn))
#define ADDS(Rd, Rm, Rn) ((Rd) = (Rm) + (Rn))
//lint -restore
#else
#define SUBS(Rd, Rm, Rn) __ASM("SUBS %0, %1, %2" : "=r"(Rd) : "r"(Rm), "r"(Rn) : "cc")
#define ADDS(Rd, Rm, Rn) __ASM("ADDS %0, %1, %2" : "=r"(Rd) : "r"(Rm), "r"(Rn) : "cc")
#endif

/**
\brief Test case: TC_CoreFunc_APSR
\details
- Check if __get_APSR instrinsic is available
- Check if __get_xPSR instrinsic is available
- Check negative, zero and overflow flags
*/
void TC_CoreFunc_APSR (void) {  
  uint32_t result;
  //lint -esym(838, Rm) unused values
  //lint -esym(438, Rm) unused values
    
  // Check negative flag
  int32_t Rm = 5;
  int32_t Rn = 7;
  SUBS(Rm, Rm, Rn);
  result  = __get_APSR();  
  ASSERT_TRUE((result & APSR_N_Msk) == APSR_N_Msk);

  Rm = 5;
  Rn = 7;
  SUBS(Rm, Rm, Rn);
  result  = __get_xPSR();  
  ASSERT_TRUE((result & xPSR_N_Msk) == xPSR_N_Msk);

  // Check zero and compare flag
  Rm = 5;
  SUBS(Rm, Rm, Rm);
  result  = __get_APSR();  
  ASSERT_TRUE((result & APSR_Z_Msk) == APSR_Z_Msk);
  ASSERT_TRUE((result & APSR_C_Msk) == APSR_C_Msk);

  Rm = 5;
  SUBS(Rm, Rm, Rm);
  result  = __get_xPSR();
  ASSERT_TRUE((result & xPSR_Z_Msk) == xPSR_Z_Msk);
  ASSERT_TRUE((result & APSR_C_Msk) == APSR_C_Msk);
  
  // Check overflow flag
  Rm = 5;
  Rn = INT32_MAX;
  ADDS(Rm, Rm, Rn);
  result  = __get_APSR(); 
  ASSERT_TRUE((result & APSR_V_Msk) == APSR_V_Msk); 

  Rm = 5;
  Rn = INT32_MAX;
  ADDS(Rm, Rm, Rn);
  result  = __get_xPSR();  
  ASSERT_TRUE((result & xPSR_V_Msk) == xPSR_V_Msk); 
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreFunc_PSP
\details
- Check if __get_PSP and __set_PSP instrinsic can be used to manipulate process stack pointer.
*/
void TC_CoreFunc_PSP (void) {  
  // don't use stack for this variables
  static uint32_t orig;
  static uint32_t psp;
  static uint32_t result;

  orig = __get_PSP();
  
  psp = orig + 0x12345678U;
  __set_PSP(psp);
  
  result = __get_PSP();
  
  __set_PSP(orig);

  ASSERT_TRUE(result == psp);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreFunc_MSP
\details
- Check if __get_MSP and __set_MSP instrinsic can be used to manipulate process stack pointer.
*/
void TC_CoreFunc_MSP (void) {  
  // don't use stack for this variables
  static uint32_t orig;
  static uint32_t msp;
  static uint32_t result;
  static uint32_t ctrl;

  ctrl = __get_CONTROL();
  __set_CONTROL(ctrl | CONTROL_SPSEL_Msk); // switch to PSP
  
  orig = __get_MSP();
  
  msp = orig + 0x12345678U;
  __set_MSP(msp);
  
  result = __get_MSP();
  
  __set_MSP(orig);

  __set_CONTROL(ctrl);
  
  ASSERT_TRUE(result == msp);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreFunc_PRIMASK
\details
- Check if __get_PRIMASK and __set_PRIMASK instrinsic can be used to manipulate PRIMASK.
- Check if __enable_irq and __disable_irq are reflected in PRIMASK.
*/
void TC_CoreFunc_PRIMASK (void) {  
  uint32_t orig = __get_PRIMASK();
  
  // toggle primask
  uint32_t primask = (orig & ~0x01U) | (~orig & 0x01U);
  
  __set_PRIMASK(primask);
  uint32_t result = __get_PRIMASK();
    
  ASSERT_TRUE(result == primask);
  
  __disable_irq();
  result = __get_PRIMASK();
  ASSERT_TRUE((result & 0x01U) == 1U);

  __enable_irq();
  result = __get_PRIMASK();
  ASSERT_TRUE((result & 0x01U) == 0U);

  __disable_irq();
  result = __get_PRIMASK();
  ASSERT_TRUE((result & 0x01U) == 1U);

  __set_PRIMASK(orig); 
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
#if ((defined (__ARM_ARCH_7M__      ) && (__ARM_ARCH_7M__      == 1)) || \
     (defined (__ARM_ARCH_7EM__     ) && (__ARM_ARCH_7EM__     == 1)) || \
     (defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1))    )

/**
\brief Test case: TC_CoreFunc_FAULTMASK
\details
- Check if __get_FAULTMASK and __set_FAULTMASK instrinsic can be used to manipulate FAULTMASK.
- Check if __enable_fault_irq and __disable_fault_irq are reflected in FAULTMASK.
*/
void TC_CoreFunc_FAULTMASK (void) {  
  uint32_t orig = __get_FAULTMASK();
  
  // toggle faultmask
  uint32_t faultmask = (orig & ~0x01U) | (~orig & 0x01U);
  
  __set_FAULTMASK(faultmask);
  uint32_t result = __get_FAULTMASK();
    
  ASSERT_TRUE(result == faultmask);
  
  __disable_fault_irq();
  result = __get_FAULTMASK();
  ASSERT_TRUE((result & 0x01U) == 1U);

  __enable_fault_irq();
  result = __get_FAULTMASK();
  ASSERT_TRUE((result & 0x01U) == 0U);

  __disable_fault_irq();
  result = __get_FAULTMASK();
  ASSERT_TRUE((result & 0x01U) == 1U);

  __set_FAULTMASK(orig);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreFunc_BASEPRI
\details
- Check if __get_BASEPRI and __set_BASEPRI instrinsic can be used to manipulate BASEPRI.
- Check if __set_BASEPRI_MAX instrinsic can be used to manipulate BASEPRI.
*/
void TC_CoreFunc_BASEPRI(void) {  
  uint32_t orig = __get_BASEPRI();
  
  uint32_t basepri = ~orig & 0x80U;
  __set_BASEPRI(basepri);
  uint32_t result = __get_BASEPRI();
  
  ASSERT_TRUE(result == basepri);
  
  __set_BASEPRI(orig);

  __set_BASEPRI_MAX(basepri);
  result = __get_BASEPRI();
  
  ASSERT_TRUE(result == basepri);
}
#endif

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
#if ((defined (__ARM_ARCH_7EM__     ) && (__ARM_ARCH_7EM__     == 1)) || \
     (defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1))    )

/**
\brief Test case: TC_CoreFunc_BASEPRI
\details
- Check if __get_FPSCR and __set_FPSCR intrinsics can be used
*/
void TC_CoreFunc_FPSCR(void) {  
  uint32_t fpscr = __get_FPSCR();
  __ISB();
  __DSB();

  __set_FPSCR(~fpscr);
  __ISB();
  __DSB();

  uint32_t result = __get_FPSCR();
    
  __set_FPSCR(fpscr);

#if (defined (__FPU_USED   ) && (__FPU_USED    == 1U))
  ASSERT_TRUE(result != fpscr);
#endif
}
#endif
