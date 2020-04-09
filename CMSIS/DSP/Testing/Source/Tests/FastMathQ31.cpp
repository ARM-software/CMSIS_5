#include "FastMathQ31.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "Test.h"


#define SNR_THRESHOLD 100
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR ((q31_t)2200)

    void FastMathQ31::test_cos_q31()
    {
        const q31_t *inp  = input.ptr();
        q31_t *refp  = ref.ptr();
        q31_t *outp  = output.ptr();
        int i;

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
        q31_t *refp  = ref.ptr();
        q31_t *outp  = output.ptr();
        int i;

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
        q31_t *refp  = ref.ptr();
        q31_t *outp  = output.ptr();
        arm_status status;
        int i;

        for(i=0; i < ref.nbSamples(); i++)
        {
           status=arm_sqrt_q31(inp[i],&outp[i]);
           ASSERT_TRUE((status == ARM_MATH_SUCCESS) || ((inp[i] <= 0) && (status == ARM_MATH_ARGUMENT_ERROR)));
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR);

    }

  
    void FastMathQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
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
        }
        
    }

    void FastMathQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      output.dump(mgr);
      
    }
