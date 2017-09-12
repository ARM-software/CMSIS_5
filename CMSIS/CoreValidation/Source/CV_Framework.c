/*-----------------------------------------------------------------------------
 *      Name:         cv_framework.c
 *      Purpose:      Test framework entry point
 *----------------------------------------------------------------------------
 *      Copyright (c) 2017 ARM Limited. All rights reserved.
 *----------------------------------------------------------------------------*/
#include "CV_Framework.h"
#include "cmsis_cv.h" 
#include "setjmp.h"

static jmp_buf jump_buffer;
  
/* Prototypes */
void ts_cmsis_cv(void);
void closeDebug(void);
void HardFault_Handler(void);
void recover(void);

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\defgroup framework_funcs Framework Functions
\brief Functions in the Framework software component
\details

@{
*/

/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/
/**
\brief Close the debug session.
\details
Debug session dead end - debug script should close session here.
*/
void closeDebug(void) {
  __NOP();
  // Test completed
}


/*=======0=========1=========2=========3=========4=========5=========6=========7=========8=========9=========0=========1====*/

/**
\brief This is CORE Validation test suite.
\details
Program flow:
  -# Test report statistics is initialized
  -# Test report headers are written to the standard output
  -# All defined test cases are executed:
      - Test case statistics is initialized
      - Test case report header is written to the standard output
      - Test case is executed
      - Test case results are written to the standard output
      - Test case report footer is written to the standard output
      - Test case is closed
  -# Test report footer is written to the standard output
  -# Debug session ends in dead loop
*/
void ts_cmsis_cv () {
  const char *fn;
  uint32_t tc, no;
  (void)ritf.Init ();                           /* Init test report                 */
  (void)ritf.Open (ts.ReportTitle,              /* Write test report title          */
                   ts.Date,                     /* Write compilation date           */
                   ts.Time,                     /* Write compilation time           */
                   ts.FileName);                /* Write module file name           */

  /* Execute all test cases */
  for (tc = 0; tc < ts.NumOfTC; tc++) {
    no = ts.TCBaseNum+tc;                 /* Test case number                 */
    fn = ts.TC[tc].TFName;                /* Test function name string        */
    (void)ritf.Open_TC (no, fn);          /* Open test case #(Base + TC)      */
    if (ts.TC[tc].en != 0U)  {
      if (setjmp(jump_buffer) == 0) {
        ts.TC[tc].TestFunc();               /* Execute test case if enabled     */
      } else {
        (void)__set_result(fn, 0U, FAILED, NULL);
      }
    }
    (void)ritf.Close_TC ();               /* Close test case                  */
  }
  (void)ritf.Close ();                    /* Close test report                */

  closeDebug();                           /* Close debug session              */
}

/**
\brief This is the entry point of the test framework.
\details
Program flow:
  -# Hardware is first initialized if Init callback function is provided
  -# Main thread is initialized
*/
void cmsis_cv (void) {
  
  /* Init test suite */
  if (ts.Init != NULL) {
    ts.Init();                           /* Init hardware                    */
  }

  ts_cmsis_cv();
}

__USED __NO_RETURN 
void recover(void) {
  longjmp(jump_buffer, 1U);
}

/** Hardfault Handler
* Catch and recover from failure states.
*
* The latest stack frame is modified so that the exception return
* results in a jump to the recovery function instead back to the
* errornous test.
*/
#if defined( __CC_ARM )
__ASM void HardFault_Handler(void) {
  IMPORT recover
  LDR R0, =recover
  STR R0, [SP, #24]
}
#else
__USED
void HardFault_Handler(void) {
  __ASM(
    "LDR R0, =recover    \n"
    "STR R0, [SP, #24]   \n" : : : "memory"
  );
}
#endif

/**
@}
*/ 
// end of group framework_funcs
