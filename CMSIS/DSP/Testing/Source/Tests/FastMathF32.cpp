#include "arm_vec_math.h"

#include "FastMathF32.h"
#include <stdio.h>

#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 119
#define SNR_ATAN2_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)
#define ABS_ERROR (1.0e-5)

#define REL_ERROR_ATAN (5.0e-7)
#define ABS_ERROR_ATAN (5.0e-7)

    void FastMathF32::test_atan2_scalar_f32()
    {
        const float32_t *inp  = input.ptr();
        float32_t *outp  = output.ptr();
        float32_t res;
        unsigned long i;
        arm_status status=ARM_MATH_SUCCESS;

        for(i=0; i < ref.nbSamples(); i++)
        {
          status=arm_atan2_f32(inp[2*i],inp[2*i+1],&res);
          outp[i]=res;
          ASSERT_TRUE((status == ARM_MATH_SUCCESS));

        }
        //printf("%f %f %f\n",inp[2*i],inp[2*i+1],outp[i]);

        //ASSERT_SNR(ref,output,(float32_t)SNR_ATAN2_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR_ATAN,REL_ERROR_ATAN);

    }
    

    void FastMathF32::test_cos_f32()
    {
        const float32_t *inp  = input.ptr();
        float32_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_cos_f32(inp[i]);
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

    void FastMathF32::test_sin_f32()
    {
        const float32_t *inp  = input.ptr();
        float32_t *outp  = output.ptr();
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
          outp[i]=arm_sin_f32(inp[i]);
        }

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

    void FastMathF32::test_sqrt_f32()
    {
        const float32_t *inp  = input.ptr();
        float32_t *outp  = output.ptr();
        arm_status status;
        unsigned long i;

        for(i=0; i < ref.nbSamples(); i++)
        {
           status=arm_sqrt_f32(inp[i],&outp[i]);
           ASSERT_TRUE((status == ARM_MATH_SUCCESS) || ((inp[i] < 0.0f) && (status == ARM_MATH_ARGUMENT_ERROR)));
        }


        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);


    }

    void FastMathF32::test_vlog_f32()
    {
        const float32_t *inp  = input.ptr();
        float32_t *outp  = output.ptr();

        arm_vlog_f32(inp,outp,ref.nbSamples());
    
        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
        ASSERT_EMPTY_TAIL(output);

    }

    void FastMathF32::test_vexp_f32()
    {
        const float32_t *inp  = input.ptr();
        float32_t *outp  = output.ptr();

        arm_vexp_f32(inp,outp,ref.nbSamples());
    
        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
        ASSERT_EMPTY_TAIL(output);

    }

  
    void FastMathF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case FastMathF32::TEST_COS_F32_1:
            {
               input.reload(FastMathF32::ANGLES1_F32_ID,mgr);
               ref.reload(FastMathF32::COS1_F32_ID,mgr);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_SIN_F32_2:
            {
               input.reload(FastMathF32::ANGLES1_F32_ID,mgr);
               ref.reload(FastMathF32::SIN1_F32_ID,mgr);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_SQRT_F32_3:
            {
               input.reload(FastMathF32::SQRTINPUT1_F32_ID,mgr);
               ref.reload(FastMathF32::SQRT1_F32_ID,mgr);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VLOG_F32_4:
            {
               input.reload(FastMathF32::LOGINPUT1_F32_ID,mgr);
               ref.reload(FastMathF32::LOG1_F32_ID,mgr);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VLOG_F32_5:
            {
               input.reload(FastMathF32::LOGINPUT1_F32_ID,mgr,3);
               ref.reload(FastMathF32::LOG1_F32_ID,mgr,3);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VLOG_F32_6:
            {
               input.reload(FastMathF32::LOGINPUT1_F32_ID,mgr,8);
               ref.reload(FastMathF32::LOG1_F32_ID,mgr,8);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VLOG_F32_7:
            {
               input.reload(FastMathF32::LOGINPUT1_F32_ID,mgr,11);
               ref.reload(FastMathF32::LOG1_F32_ID,mgr,11);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VEXP_F32_8:
            {
               input.reload(FastMathF32::EXPINPUT1_F32_ID,mgr);
               ref.reload(FastMathF32::EXP1_F32_ID,mgr);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VEXP_F32_9:
            {
               input.reload(FastMathF32::EXPINPUT1_F32_ID,mgr,3);
               ref.reload(FastMathF32::EXP1_F32_ID,mgr,3);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VEXP_F32_10:
            {
               input.reload(FastMathF32::EXPINPUT1_F32_ID,mgr,8);
               ref.reload(FastMathF32::EXP1_F32_ID,mgr,8);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_VEXP_F32_11:
            {
               input.reload(FastMathF32::EXPINPUT1_F32_ID,mgr,11);
               ref.reload(FastMathF32::EXP1_F32_ID,mgr,11);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);

            }
            break;

            case FastMathF32::TEST_ATAN2_SCALAR_F32_12:
            {
               input.reload(FastMathF32::ATAN2INPUT1_F32_ID,mgr);
               ref.reload(FastMathF32::ATAN2_F32_ID,mgr);
               output.create(ref.nbSamples(),FastMathF32::OUT_F32_ID,mgr);
            }
            break;

           

        }
        
    }

    void FastMathF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
