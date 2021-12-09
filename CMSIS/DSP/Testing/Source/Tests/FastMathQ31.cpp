#include "FastMathQ31.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 100
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_SQRT_ERROR ((q31_t)7)

#define ABS_ERROR ((q31_t)2200)
#define ABS_DIV_ERROR ((q31_t)1)


#define LOG_ABS_ERROR ((q31_t)2)


    void FastMathQ31::test_vlog_q31()
    {
        const q31_t *inp  = input.ptr();
        q31_t *outp  = output.ptr();

        arm_vlog_q31(inp,outp,ref.nbSamples());
        
        
        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,LOG_ABS_ERROR);
        ASSERT_EMPTY_TAIL(output);

    }

    void FastMathQ31::test_division_q31()
    {
        const q31_t *nump  = numerator.ptr();
        const q31_t *denp  = denominator.ptr();
        q31_t *outp  = output.ptr();
        int16_t *shiftp  = shift.ptr();
        arm_status status;

      
        for(unsigned long i=0; i < ref.nbSamples(); i++)
        {

          status = arm_divide_q31(nump[i],denp[i],&outp[i],&shiftp[i]);
        }

        (void)status;

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_DIV_ERROR);
        ASSERT_EQ(refShift,shift);

    }

    void FastMathQ31::test_cos_q31()
    {
        const q31_t *inp  = input.ptr();
        q31_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_cos_q31(inp[i]);
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR);

    }

    void FastMathQ31::test_sin_q31()
    {
        const q31_t *inp  = input.ptr();
        q31_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_sin_q31(inp[i]);
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR);

    }

    void FastMathQ31::test_sqrt_q31()
    {
        const q31_t *inp  = input.ptr();
        q31_t *outp  = output.ptr();
        arm_status status;
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
           status=arm_sqrt_q31(inp[i],&outp[i]);
           ASSERT_TRUE((status == ARM_MATH_SUCCESS) || ((inp[i] <= 0) && (status == ARM_MATH_ARGUMENT_ERROR)));
        }

        //ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_SQRT_ERROR);

    }

  
    void FastMathQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case FastMathQ31::TEST_COS_Q31_1:
            {
               input.reload(FastMathQ31::ANGLES1_Q31_ID,mgr);
               ref.reload(FastMathQ31::COS1_Q31_ID,mgr);
               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);

            }
            break;

            case FastMathQ31::TEST_SIN_Q31_2:
            {
               input.reload(FastMathQ31::ANGLES1_Q31_ID,mgr);
               ref.reload(FastMathQ31::SIN1_Q31_ID,mgr);
               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);

            }
            break;

            case FastMathQ31::TEST_SQRT_Q31_3:
            {
               input.reload(FastMathQ31::SQRTINPUT1_Q31_ID,mgr);
               ref.reload(FastMathQ31::SQRT1_Q31_ID,mgr);
               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);

            }
            break;

            case FastMathQ31::TEST_DIVISION_Q31_4:
            {
               numerator.reload(FastMathQ31::NUMERATOR_Q31_ID,mgr);
               denominator.reload(FastMathQ31::DENOMINATOR_Q31_ID,mgr);

               ref.reload(FastMathQ31::DIVISION_VALUE_Q31_ID,mgr);
               refShift.reload(FastMathQ31::DIVISION_SHIFT_S16_ID,mgr);

               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);
               shift.create(ref.nbSamples(),FastMathQ31::SHIFT_S16_ID,mgr);

            }
            break;

            case FastMathQ31::TEST_VLOG_Q31_5:
            {
               input.reload(FastMathQ31::LOGINPUT1_Q31_ID,mgr);
               ref.reload(FastMathQ31::LOG1_Q31_ID,mgr);
               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);

            }
            break;

            case FastMathQ31::TEST_VLOG_Q31_6:
            {
               input.reload(FastMathQ31::LOGINPUT1_Q31_ID,mgr,3);
               ref.reload(FastMathQ31::LOG1_Q31_ID,mgr,3);
               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);

            }
            break;

            case FastMathQ31::TEST_VLOG_Q31_7:
            {
               input.reload(FastMathQ31::LOGINPUT1_Q31_ID,mgr,8);
               ref.reload(FastMathQ31::LOG1_Q31_ID,mgr,8);
               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);

            }
            break;

            case FastMathQ31::TEST_VLOG_Q31_8:
            {
               input.reload(FastMathQ31::LOGINPUT1_Q31_ID,mgr,11);
               ref.reload(FastMathQ31::LOG1_Q31_ID,mgr,11);
               output.create(ref.nbSamples(),FastMathQ31::OUT_Q31_ID,mgr);

            }
            break;
        }
        
    }

    void FastMathQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
