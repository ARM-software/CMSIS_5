#include "arm_vec_math.h"

#include "FastMathF64.h"
#include <stdio.h>

#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 310
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (2.0e-16)
#define ABS_ERROR (2.0e-16)

/*
    void FastMathF64::test_cos_f64()
    {
        const float64_t *inp  = input.ptr();
        float64_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_cos_f64(inp[i]);
        }

        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

    void FastMathF64::test_sin_f64()
    {
        const float64_t *inp  = input.ptr();
        float64_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_sin_f64(inp[i]);
        }

        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

    void FastMathF64::test_sqrt_f64()
    {
        const float64_t *inp  = input.ptr();
        float64_t *outp  = output.ptr();
        arm_status status;
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
           status=arm_sqrt_f64(inp[i],&outp[i]);
           ASSERT_TRUE((status == ARM_MATH_SUCCESS) || ((inp[i] < 0.0f) && (status == ARM_MATH_ARGUMENT_ERROR)));
        }


        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);


    }

*/
    void FastMathF64::test_vlog_f64()
    {
        const float64_t *inp  = input.ptr();
        float64_t *outp  = output.ptr();

        arm_vlog_f64(inp,outp,ref.nbSamples());
    
        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
        ASSERT_EMPTY_TAIL(output);

    }

    void FastMathF64::test_vexp_f64()
    {
        const float64_t *inp  = input.ptr();
        float64_t *outp  = output.ptr();

        arm_vexp_f64(inp,outp,ref.nbSamples());
    
        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
        ASSERT_EMPTY_TAIL(output);

    }

  
    void FastMathF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case FastMathF64::TEST_COS_F64_1:
            {
               input.reload(FastMathF64::ANGLES1_F64_ID,mgr);
               ref.reload(FastMathF64::COS1_F64_ID,mgr);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_SIN_F64_2:
            {
               input.reload(FastMathF64::ANGLES1_F64_ID,mgr);
               ref.reload(FastMathF64::SIN1_F64_ID,mgr);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_SQRT_F64_3:
            {
               input.reload(FastMathF64::SQRTINPUT1_F64_ID,mgr);
               ref.reload(FastMathF64::SQRT1_F64_ID,mgr);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VLOG_F64_4:
            {
               input.reload(FastMathF64::LOGINPUT1_F64_ID,mgr);
               ref.reload(FastMathF64::LOG1_F64_ID,mgr);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VLOG_F64_5:
            {
               /*
                  If only one sample was taken here, the SNR
                  computation would give 0 / 0 because the
                  first value (1.0) has a log of 0.

               */
               input.reload(FastMathF64::LOGINPUT1_F64_ID,mgr,2);
               ref.reload(FastMathF64::LOG1_F64_ID,mgr,2);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VLOG_F64_6:
            {
               input.reload(FastMathF64::LOGINPUT1_F64_ID,mgr,4);
               ref.reload(FastMathF64::LOG1_F64_ID,mgr,4);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VLOG_F64_7:
            {
               input.reload(FastMathF64::LOGINPUT1_F64_ID,mgr,5);
               ref.reload(FastMathF64::LOG1_F64_ID,mgr,5);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VEXP_F64_8:
            {
               input.reload(FastMathF64::EXPINPUT1_F64_ID,mgr);
               ref.reload(FastMathF64::EXP1_F64_ID,mgr);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VEXP_F64_9:
            {
               input.reload(FastMathF64::EXPINPUT1_F64_ID,mgr,2);
               ref.reload(FastMathF64::EXP1_F64_ID,mgr,2);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VEXP_F64_10:
            {
               input.reload(FastMathF64::EXPINPUT1_F64_ID,mgr,4);
               ref.reload(FastMathF64::EXP1_F64_ID,mgr,4);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;

            case FastMathF64::TEST_VEXP_F64_11:
            {
               input.reload(FastMathF64::EXPINPUT1_F64_ID,mgr,5);
               ref.reload(FastMathF64::EXP1_F64_ID,mgr,5);
               output.create(ref.nbSamples(),FastMathF64::OUT_F64_ID,mgr);

            }
            break;
        }
        
    }

    void FastMathF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
