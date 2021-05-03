#include "FastMathQ15.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



#define SNR_THRESHOLD 70
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR ((q15_t)10)



    void FastMathQ15::test_division_q15()
    {
        const q15_t *nump  = numerator.ptr();
        const q15_t *denp  = denominator.ptr();
        q15_t *outp  = output.ptr();
        int16_t *shiftp  = shift.ptr();
        arm_status status;

      
        for(unsigned long i=0; i < ref.nbSamples(); i++)
        {
          status = arm_divide_q15(nump[i],denp[i],&outp[i],&shiftp[i]);
        }

        (void)status;

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR);
        ASSERT_EQ(refShift,shift);

    }


    void FastMathQ15::test_cos_q15()
    {
        const q15_t *inp  = input.ptr();
        q15_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_cos_q15(inp[i]);
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR);

    }

    void FastMathQ15::test_sin_q15()
    {
        const q15_t *inp  = input.ptr();
        q15_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_sin_q15(inp[i]);
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR);

    }

    void FastMathQ15::test_sqrt_q15()
    {
        const q15_t *inp  = input.ptr();
        q15_t *outp  = output.ptr();
        arm_status status;
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
           status=arm_sqrt_q15(inp[i],&outp[i]);
           ASSERT_TRUE((status == ARM_MATH_SUCCESS) || ((inp[i] <= 0) && (status == ARM_MATH_ARGUMENT_ERROR)));
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR);

    }

  
    void FastMathQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case FastMathQ15::TEST_COS_Q15_1:
            {
               input.reload(FastMathQ15::ANGLES1_Q15_ID,mgr);
               ref.reload(FastMathQ15::COS1_Q15_ID,mgr);
               output.create(ref.nbSamples(),FastMathQ15::OUT_Q15_ID,mgr);

            }
            break;

            case FastMathQ15::TEST_SIN_Q15_2:
            {
               input.reload(FastMathQ15::ANGLES1_Q15_ID,mgr);
               ref.reload(FastMathQ15::SIN1_Q15_ID,mgr);
               output.create(ref.nbSamples(),FastMathQ15::OUT_Q15_ID,mgr);

            }
            break;

            case FastMathQ15::TEST_SQRT_Q15_3:
            {
               input.reload(FastMathQ15::SQRTINPUT1_Q15_ID,mgr);
               ref.reload(FastMathQ15::SQRT1_Q15_ID,mgr);
               output.create(ref.nbSamples(),FastMathQ15::OUT_Q15_ID,mgr);

            }
            break;

            case FastMathQ15::TEST_DIVISION_Q15_4:
            {
               numerator.reload(FastMathQ15::NUMERATOR_Q15_ID,mgr);
               denominator.reload(FastMathQ15::DENOMINATOR_Q15_ID,mgr);

               ref.reload(FastMathQ15::DIVISION_VALUE_Q15_ID,mgr);
               refShift.reload(FastMathQ15::DIVISION_SHIFT_S16_ID,mgr);

               output.create(ref.nbSamples(),FastMathQ15::OUT_Q15_ID,mgr);
               shift.create(ref.nbSamples(),FastMathQ15::SHIFT_S16_ID,mgr);

            }
            break;
        }
        
    }

    void FastMathQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
