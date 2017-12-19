#ifndef _MAIN_H_
#define _MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef MBED
#include "mbed.h"
#endif  /*  */
#include "arm_math.h"
    
#include "arm_nnfunctions.h"
#include "ref_functions.h"
void verify_results_q7(q7_t * ref, q7_t * opt, int length)
{
    bool if_match = true;
    for (int i = 0; i < length; i++)
    {
        if (ref[i] != opt[i])
        {
            printf("Output mismatch at %d, expected %d, actual %d\r\n", i, ref[i], opt[i]);
            if_match = false;
        }
    }
    if (if_match == true)
    {
        printf("Outputs match.\r\n\r\n");
    }
 }

 void verify_results_q15(q15_t * ref, q15_t * opt, int length)
{
    bool if_match = true;
    for (int i = 0; i < length; i++)
    {
        if (ref[i] != opt[i])
        {
            printf("Output mismatch at %d, expected %d, actual %d\r\n", i, ref[i], opt[i]);
            if_match = false;
        }
    }
    if (if_match == true)
    {
        printf("Outputs match.\r\n\r\n");
    }
 }

  
#endif  /*  */
