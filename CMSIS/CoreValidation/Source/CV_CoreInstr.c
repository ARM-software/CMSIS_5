/*-----------------------------------------------------------------------------
 *      Name:         CV_CoreInstr.c 
 *      Purpose:      CMSIS CORE validation tests implementation
 *-----------------------------------------------------------------------------
 *      Copyright (c) 2017 ARM Limited. All rights reserved.
 *----------------------------------------------------------------------------*/

#include "CV_Framework.h"
#include "cmsis_cv.h"

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
}

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Test case: TC_CoreInstr_REV16
\details
- Check if __REV16 instrinsic swaps the bytes in a halfword.
*/
void TC_CoreInstr_REV16(void) {
  uint16_t result = __REV16(0x4711U);
  ASSERT_TRUE(result == 0x1147U);

  result = __REV16(0x4711U);
  ASSERT_TRUE(result == 0x1147U);
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

  result = __REVSH(-4711);
  ASSERT_TRUE(result == -26131);
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
  int32_t result = __CLZ(0x00U);
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
