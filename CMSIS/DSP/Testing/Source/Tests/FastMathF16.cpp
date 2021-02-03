#include "FastMathF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 60
#define SNR_LOG_THRESHOLD 40

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-3)
#define ABS_ERROR (1.0e-3)

#define REL_LOG_ERROR (3.0e-2)
#define ABS_LOG_ERROR (3.0e-2)

#if 0
    void FastMathF16::test_cos_f16()
    {
        const float16_t *inp  = input.ptr();
        float16_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_cos_f16(inp[i]);
        }

        ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

    void FastMathF16::test_sin_f16()
    {
        const float16_t *inp  = input.ptr();
        float16_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_sin_f16(inp[i]);
        }

        ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

#endif 

    void FastMathF16::test_sqrt_f16()
    {
        const float16_t *inp  = input.ptr();
        float16_t *outp  = output.ptr();
        arm_status status;
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
           status=arm_sqrt_f16(inp[i],&outp[i]);
           ASSERT_TRUE((status == ARM_MATH_SUCCESS) || ((inp[i] < 0.0f) && (status == ARM_MATH_ARGUMENT_ERROR)));
        }


        ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);


    }

    void FastMathF16::test_vlog_f16()
    {
        const float16_t *inp  = input.ptr();
        float16_t *outp  = output.ptr();

        arm_vlog_f16(inp,outp,ref.nbSamples());
    
        ASSERT_SNR(ref,output,(float16_t)SNR_LOG_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_LOG_ERROR,REL_LOG_ERROR);
        ASSERT_EMPTY_TAIL(output);

    }

    void FastMathF16::test_vexp_f16()
    {
        const float16_t *inp  = input.ptr();
        float16_t *outp  = output.ptr();

        arm_vexp_f16(inp,outp,ref.nbSamples());
    
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
        ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(output);

    }

    void FastMathF16::test_inverse_f16()
    {
        const float16_t *inp  = input.ptr();

        float16_t *outp  = output.ptr();

        arm_vinverse_f16(inp,outp,ref.nbSamples());

        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
        ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(output);

    }

  
    void FastMathF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
#if 0
            case FastMathF16::TEST_COS_F16_1:
            {
               input.reload(FastMathF16::ANGLES1_F16_ID,mgr);
               ref.reload(FastMathF16::COS1_F16_ID,mgr);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_SIN_F16_2:
            {
               input.reload(FastMathF16::ANGLES1_F16_ID,mgr);
               ref.reload(FastMathF16::SIN1_F16_ID,mgr);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;
#endif 
          
            case FastMathF16::TEST_SQRT_F16_3:
            {
               input.reload(FastMathF16::SQRTINPUT1_F16_ID,mgr);
               ref.reload(FastMathF16::SQRT1_F16_ID,mgr);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VLOG_F16_4:
            {
               input.reload(FastMathF16::LOGINPUT1_F16_ID,mgr);
               ref.reload(FastMathF16::LOG1_F16_ID,mgr);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VLOG_F16_5:
            {
               input.reload(FastMathF16::LOGINPUT1_F16_ID,mgr,7);
               ref.reload(FastMathF16::LOG1_F16_ID,mgr,7);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VLOG_F16_6:
            {
               input.reload(FastMathF16::LOGINPUT1_F16_ID,mgr,16);
               ref.reload(FastMathF16::LOG1_F16_ID,mgr,16);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VLOG_F16_7:
            {
               input.reload(FastMathF16::LOGINPUT1_F16_ID,mgr,23);
               ref.reload(FastMathF16::LOG1_F16_ID,mgr,23);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VEXP_F16_8:
            {
              
              input.reload(FastMathF16::EXPINPUT1_F16_ID,mgr);
              ref.reload(FastMathF16::EXP1_F16_ID,mgr);
              output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VEXP_F16_9:
            {
               input.reload(FastMathF16::EXPINPUT1_F16_ID,mgr,7);
               ref.reload(FastMathF16::EXP1_F16_ID,mgr,7);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VEXP_F16_10:
            {
               input.reload(FastMathF16::EXPINPUT1_F16_ID,mgr,16);
               ref.reload(FastMathF16::EXP1_F16_ID,mgr,16);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_VEXP_F16_11:
            {
               input.reload(FastMathF16::EXPINPUT1_F16_ID,mgr,23);
               ref.reload(FastMathF16::EXP1_F16_ID,mgr,23);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;

            case FastMathF16::TEST_INVERSE_F16_12:
            {
               input.reload(FastMathF16::INPUT1_F16_ID,mgr);
               ref.reload(FastMathF16::INVERSE1_F16_ID,mgr);
               output.create(ref.nbSamples(),FastMathF16::OUT_F16_ID,mgr);

            }
            break;
        }
        
    }

    void FastMathF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
