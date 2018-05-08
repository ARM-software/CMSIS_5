/*-----------------------------------------------------------------------------
 *      Name:         CV_CoreInstr.c 
 *      Purpose:      CMSIS CORE validation tests implementation
 *-----------------------------------------------------------------------------
 *      Copyright (c) 2017 ARM Limited. All rights reserved.
 *----------------------------------------------------------------------------*/

#include "CV_Framework.h"
#include "cmsis_cv.h"

#if defined(__CORTEX_M)
#elif defined(__CORTEX_A)
#include "irq_ctrl.h"
#else
#error __CORTEX_M or __CORTEX_A must be defined!
#endif

/*-----------------------------------------------------------------------------
 *      Test implementation
 *----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
 *      Test cases
 *----------------------------------------------------------------------------*/

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_NOP
\details
- Check if __NOP instrinsic is available
- No real assertion is deployed, just a compile time check.
*/
void TC_CoreInstr_NOP (void) {
  __NOP();
  ASSERT_TRUE(1U == 1U);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_REV
\details
- Check if __REV instrinsic swaps all bytes in a word.
*/
void TC_CoreInstr_REV (void) {
  uint32_t result = __REV(0x47110815U);
  ASSERT_TRUE(result == 0x15081147U);
  
  result = __REV(0x80000000U);
  ASSERT_TRUE(result == 0x00000080U);

  result = __REV(0x00000080U);
  ASSERT_TRUE(result == 0x80000000U);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_REV16
\details
- Check if __REV16 instrinsic swaps the bytes in both halfwords independendly.
*/
void TC_CoreInstr_REV16(void) {
  uint32_t result = __REV16(0x47110815U);
  ASSERT_TRUE(result == 0x11471508U); 
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_REVSH
\details
- Check if __REVSH instrinsic swaps bytes in a signed halfword keeping the sign.
*/
void TC_CoreInstr_REVSH(void) {
  int16_t result = __REVSH(0x4711);
  ASSERT_TRUE(result == 0x1147);

  result = __REVSH((int16_t)0x8000);
  ASSERT_TRUE(result == 0x0080);

  result = __REVSH(0x0080);
  ASSERT_TRUE(result == (int16_t)0x8000);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_ROT
\details
- Check if __ROR instrinsic moves all bits as expected.
*/
void TC_CoreInstr_ROR(void) {
  uint32_t result = __ROR(0x01U, 1U);
  ASSERT_TRUE(result == 0x80000000U);

  result = __ROR(0x80000000U, 1U);
  ASSERT_TRUE(result == 0x40000000U);

  result = __ROR(0x40000000U, 30U);
  ASSERT_TRUE(result == 0x00000001U);

  result = __ROR(0x01U, 32U);
  ASSERT_TRUE(result == 0x00000001U);

  result = __ROR(0x08154711U, 8U);
  ASSERT_TRUE(result == 0x11081547U);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_RBIT
\details
- Check if __RBIT instrinsic revserses the bit order of arbitrary words.
*/
void TC_CoreInstr_RBIT (void) {
  uint32_t result = __RBIT(0xAAAAAAAAU);
  ASSERT_TRUE(result == 0x55555555U);

  result = __RBIT(0x55555555U);
  ASSERT_TRUE(result == 0xAAAAAAAAU);
  
  result = __RBIT(0x00000001U);
  ASSERT_TRUE(result == 0x80000000U);
  
  result = __RBIT(0x80000000U); 
  ASSERT_TRUE(result == 0x00000001U);
  
  result = __RBIT(0xDEADBEEFU); 
  ASSERT_TRUE(result == 0xF77DB57BU);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_CLZ
\details
- Check if __CLZ instrinsic counts leading zeros.
*/
void TC_CoreInstr_CLZ (void) {
  uint32_t result = __CLZ(0x00U);
  ASSERT_TRUE(result == 32);

  result = __CLZ(0x00000001U);
  ASSERT_TRUE(result == 31);

  result = __CLZ(0x40000000U);
  ASSERT_TRUE(result == 1);

  result = __CLZ(0x80000000U);
  ASSERT_TRUE(result == 0);

  result = __CLZ(0xFFFFFFFFU);
  ASSERT_TRUE(result == 0);

  result = __CLZ(0x80000001U);
  ASSERT_TRUE(result == 0);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
#if ((defined (__ARM_ARCH_7M__      ) && (__ARM_ARCH_7M__      == 1)) || \
     (defined (__ARM_ARCH_7EM__     ) && (__ARM_ARCH_7EM__     == 1)) || \
     (defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) || \
     (defined (__ARM_ARCH_8M_BASE__ ) && (__ARM_ARCH_8M_BASE__ == 1)) || \
     (defined(__CORTEX_A)                                           )    )
  
/// Exclusive byte value
static volatile uint8_t TC_CoreInstr_Exclusives_byte = 0x47U;

/// Exclusive halfword value
static volatile uint16_t TC_CoreInstr_Exclusives_hword = 0x0815U;

/// Exclusive word value
static volatile uint32_t TC_CoreInstr_Exclusives_word = 0x08154711U;

/** 
\brief Interrupt function for TC_CoreInstr_Exclusives
\details
The interrupt manipulates all the global data
which disrupts the exclusive sequences in the test
*/
static void TC_CoreInstr_ExclusivesIRQHandler(void) {
  const uint8_t b = __LDREXB(&TC_CoreInstr_Exclusives_byte);
  __STREXB((uint8_t)~b, &TC_CoreInstr_Exclusives_byte);
  const uint16_t hw = __LDREXH(&TC_CoreInstr_Exclusives_hword);
  __STREXH((uint16_t)~hw, &TC_CoreInstr_Exclusives_hword);
  const uint32_t w = __LDREXW(&TC_CoreInstr_Exclusives_word);
  __STREXW((uint32_t)~w, &TC_CoreInstr_Exclusives_word);
}

/** 
\brief Helper function for TC_CoreInstr_Exclusives to enable test interrupt.
\details
This helper function implements interrupt enabling according to target
architecture, i.e. Cortex-A or Cortex-M.
*/
static void TC_CoreInstr_ExclusivesIRQEnable(void) {
#if defined(__CORTEX_M)
  TST_IRQHandler = TC_CoreInstr_ExclusivesIRQHandler;
  NVIC_EnableIRQ(WDT_IRQn);
#elif defined(__CORTEX_A)
  IRQ_SetHandler(SGI0_IRQn, TC_CoreInstr_ExclusivesIRQHandler);
  IRQ_Enable(SGI0_IRQn);
#else
  #error __CORTEX_M or __CORTEX_A must be defined!
#endif
  __enable_irq();
}

/** 
\brief Helper function for TC_CoreInstr_Exclusives to set test interrupt pending.
\details
This helper function implements set pending the test interrupt according to target
architecture, i.e. Cortex-A or Cortex-M.
*/
static void TC_CoreInstr_ExclusivesIRQPend(void) {
#if defined(__CORTEX_M)
  NVIC_SetPendingIRQ(WDT_IRQn);
#elif defined(__CORTEX_A)
  IRQ_SetPending(SGI0_IRQn);
#else
  #error __CORTEX_M or __CORTEX_A must be defined!
#endif
  for(uint32_t i = 10U; i > 0U; --i) {}
}

/** 
\brief Helper function for TC_CoreInstr_Exclusives to disable test interrupt.
\details
This helper function implements interrupt disabling according to target
architecture, i.e. Cortex-A or Cortex-M.
*/
static void TC_CoreInstr_ExclusivesIRQDisable(void) {
  __disable_irq();
#if defined(__CORTEX_M)
  NVIC_DisableIRQ(WDT_IRQn);
  TST_IRQHandler = NULL;
#elif defined(__CORTEX_A)
  IRQ_Disable(SGI0_IRQn);
  IRQ_SetHandler(SGI0_IRQn, NULL);
#else
  #error __CORTEX_M or __CORTEX_A must be defined!
#endif
}

/**
\brief Test case: TC_CoreInstr_Exclusives
\details
Checks exclusive load and store instructions:
- LDREXB, LDREXH, LDREXW
- STREXB, STREXH, STREXW
- CLREX
*/
void TC_CoreInstr_Exclusives (void) {
  /* 1. Test exclusives without interruption */
    do {
      const uint8_t v = __LDREXB(&TC_CoreInstr_Exclusives_byte);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_byte);
      
      const uint32_t result = __STREXB(v+1U, &TC_CoreInstr_Exclusives_byte);
      ASSERT_TRUE(result == 0U);
      ASSERT_TRUE(TC_CoreInstr_Exclusives_byte == v+1U);
    } while(0);
    
    do {
     const uint16_t v = __LDREXH(&TC_CoreInstr_Exclusives_hword);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_hword);
      
      const uint32_t result = __STREXH(v+1U, &TC_CoreInstr_Exclusives_hword);
      ASSERT_TRUE(result == 0U);
      ASSERT_TRUE(TC_CoreInstr_Exclusives_hword == v+1U);
    } while(0);
      
    do {
      const uint32_t v = __LDREXW(&TC_CoreInstr_Exclusives_word);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_word);
      
      const uint32_t result = __STREXW(v+1U, &TC_CoreInstr_Exclusives_word);
      ASSERT_TRUE(result == 0U);
      ASSERT_TRUE(TC_CoreInstr_Exclusives_word == v+1U);
    } while(0);
  
  /* 2. Test exclusives with clear */
    do {
      const uint8_t v = __LDREXB(&TC_CoreInstr_Exclusives_byte);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_byte);
      
      __CLREX();
      
      const uint32_t result = __STREXB(v+1U, &TC_CoreInstr_Exclusives_byte);
      ASSERT_TRUE(result == 1U);
      ASSERT_TRUE(TC_CoreInstr_Exclusives_byte == v);
    } while(0);
    
    do {
      const uint16_t v = __LDREXH(&TC_CoreInstr_Exclusives_hword);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_hword);
      
      __CLREX();
      
      const uint32_t result = __STREXH(v+1U, &TC_CoreInstr_Exclusives_hword);
      ASSERT_TRUE(result == 1U);
      ASSERT_TRUE(TC_CoreInstr_Exclusives_hword == v);
    } while(0);
      
    do {
      const uint32_t v = __LDREXW(&TC_CoreInstr_Exclusives_word);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_word);
      
      __CLREX();
      
      const uint32_t result = __STREXW(v+1U, &TC_CoreInstr_Exclusives_word);
      ASSERT_TRUE(result == 1U);
      ASSERT_TRUE(TC_CoreInstr_Exclusives_word == v);
    } while(0);
    
  /* 3. Test exclusives with interruption */
    
    TC_CoreInstr_ExclusivesIRQEnable();
    
    do {
      const uint8_t v = __LDREXB(&TC_CoreInstr_Exclusives_byte);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_byte);
      
      TC_CoreInstr_ExclusivesIRQPend();
        
      const uint32_t result = __STREXB(v+1U, &TC_CoreInstr_Exclusives_byte);
      ASSERT_TRUE(result == 1U);
      
      const uint8_t iv = ~v;
      ASSERT_TRUE(iv == TC_CoreInstr_Exclusives_byte);
    } while(0);
    
    do {
      const uint16_t v = __LDREXH(&TC_CoreInstr_Exclusives_hword);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_hword);
      
      TC_CoreInstr_ExclusivesIRQPend();
      
      const uint32_t result = __STREXH(v+1U, &TC_CoreInstr_Exclusives_hword);
      ASSERT_TRUE(result == 1U);
      
      const uint16_t iv = ~v;
      ASSERT_TRUE(iv == TC_CoreInstr_Exclusives_hword);
    } while(0);
      
    do {
      const uint32_t v = __LDREXW(&TC_CoreInstr_Exclusives_word);
      ASSERT_TRUE(v == TC_CoreInstr_Exclusives_word);
      
      TC_CoreInstr_ExclusivesIRQPend();
        
      const uint32_t result = __STREXW(v+1U, &TC_CoreInstr_Exclusives_word);
      ASSERT_TRUE(result == 1U);
      
      const uint32_t iv = ~v;
      ASSERT_TRUE(iv == TC_CoreInstr_Exclusives_word);
    } while(0);
    
    TC_CoreInstr_ExclusivesIRQDisable();
}
#endif

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_SSAT
\details
- Check if __SSAT instrinsic saturates signed integer values.
*/
void TC_CoreInstr_SSAT (void) {
  int32_t result = __SSAT(INT32_MAX, 32U);
  ASSERT_TRUE(result == INT32_MAX);

  result = __SSAT(INT32_MAX, 16U);
  ASSERT_TRUE(result == INT16_MAX);

  result = __SSAT(INT32_MAX, 8U);
  ASSERT_TRUE(result == INT8_MAX);
 
  result = __SSAT(INT32_MAX, 1U);
  ASSERT_TRUE(result == 0);

  result = __SSAT(INT32_MIN, 32U);
  ASSERT_TRUE(result == INT32_MIN);

  result = __SSAT(INT32_MIN, 16U);
  ASSERT_TRUE(result == INT16_MIN);

  result = __SSAT(INT32_MIN, 8U);
  ASSERT_TRUE(result == INT8_MIN);

  result = __SSAT(INT32_MIN, 1U);
  ASSERT_TRUE(result == -1);
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_USAT
\details
- Check if __USAT instrinsic saturates unsigned integer values.
*/
void TC_CoreInstr_USAT (void) {
  uint32_t result = __USAT(INT32_MAX, 31U);
  ASSERT_TRUE(result == (UINT32_MAX>>1U));

  result = __USAT(INT32_MAX, 16U);
  ASSERT_TRUE(result == UINT16_MAX);

  result = __USAT(INT32_MAX, 8U);
  ASSERT_TRUE(result == UINT8_MAX);

  result = __USAT(INT32_MAX, 0U);
  ASSERT_TRUE(result == 0U);

  result = __USAT(INT32_MIN, 31U);
  ASSERT_TRUE(result == 0U);

  result = __USAT(INT32_MIN, 16U);
  ASSERT_TRUE(result == 0U);

  result = __USAT(INT32_MIN, 8U);
  ASSERT_TRUE(result == 0U);

  result = __USAT(INT32_MIN, 0U);
  ASSERT_TRUE(result == 0U);
}
