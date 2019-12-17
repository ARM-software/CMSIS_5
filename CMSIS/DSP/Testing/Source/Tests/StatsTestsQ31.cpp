#include "arm_math.h"
#include "StatsTestsQ31.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

//#include <cstdio>

#define SNR_THRESHOLD 100
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)(100))
#define ABS_ERROR_Q63 ((q63_t)(1<<18))

    void StatsTestsQ31::test_max_q31()
    {
        const q31_t *inp  = inputA.ptr();

        q31_t result;
        uint32_t  indexval;

        q31_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        q31_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_max_q31(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsQ31::test_min_q31()
    {
        const q31_t *inp  = inputA.ptr();

        q31_t result;
        uint32_t  indexval;

        q31_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        q31_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_min_q31(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsQ31::test_mean_q31()
    {
        const q31_t *inp  = inputA.ptr();

        q31_t result;

        q31_t *refp  = ref.ptr();

        q31_t *outp  = output.ptr();

        arm_mean_q31(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q31);

    }

    void StatsTestsQ31::test_power_q31()
    {
        const q31_t *inp  = inputA.ptr();

        q63_t result;

        q63_t *refp  = refPower.ptr();

        q63_t *outp  = outputPower.ptr();

        arm_power_q31(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],(q63_t)ABS_ERROR_Q63);

    }

    void StatsTestsQ31::test_rms_q31()
    {
        const q31_t *inp  = inputA.ptr();

        q31_t result;

        q31_t *refp  = ref.ptr();

        q31_t *outp  = output.ptr();

        arm_rms_q31(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q31);

    }

    void StatsTestsQ31::test_std_q31()
    {
        const q31_t *inp  = inputA.ptr();

        q31_t result;

        q31_t *refp  = ref.ptr();

        q31_t *outp  = output.ptr();

        arm_std_q31(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q31);

    }

    void StatsTestsQ31::test_var_q31()
    {
        const q31_t *inp  = inputA.ptr();

        q31_t result;

        q31_t *refp  = ref.ptr();

        q31_t *outp  = output.ptr();

        arm_var_q31(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q31);

    }

  
  
    void StatsTestsQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        switch(id)
        {
            case StatsTestsQ31::TEST_MAX_Q31_1:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,3);
              
               maxIndexes.reload(StatsTestsQ31::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ31::MAXVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);
               index.create(1,StatsTestsQ31::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ31::TEST_MAX_Q31_2:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,8);
              
               maxIndexes.reload(StatsTestsQ31::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ31::MAXVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);
               index.create(1,StatsTestsQ31::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ31::TEST_MAX_Q31_3:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,11);
              
               maxIndexes.reload(StatsTestsQ31::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ31::MAXVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);
               index.create(1,StatsTestsQ31::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ31::TEST_MEAN_Q31_4:
            {
               inputA.reload(StatsTestsQ31::INPUT2_Q31_ID,mgr,3);
              
               ref.reload(StatsTestsQ31::MEANVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ31::TEST_MEAN_Q31_5:
            {
               inputA.reload(StatsTestsQ31::INPUT2_Q31_ID,mgr,8);
              
               ref.reload(StatsTestsQ31::MEANVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ31::TEST_MEAN_Q31_6:
            {
               inputA.reload(StatsTestsQ31::INPUT2_Q31_ID,mgr,11);
              
               ref.reload(StatsTestsQ31::MEANVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ31::TEST_MIN_Q31_7:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,3);
              
               minIndexes.reload(StatsTestsQ31::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ31::MINVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);
               index.create(1,StatsTestsQ31::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ31::TEST_MIN_Q31_8:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,8);
              
               minIndexes.reload(StatsTestsQ31::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ31::MINVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);
               index.create(1,StatsTestsQ31::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ31::TEST_MIN_Q31_9:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,11);
              
               minIndexes.reload(StatsTestsQ31::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ31::MINVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);
               index.create(1,StatsTestsQ31::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ31::TEST_POWER_Q31_10:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,3);
              
               refPower.reload(StatsTestsQ31::POWERVALS_Q63_ID,mgr);
               
               outputPower.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ31::TEST_POWER_Q31_11:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,8);
              
               refPower.reload(StatsTestsQ31::POWERVALS_Q63_ID,mgr);
               
               outputPower.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ31::TEST_POWER_Q31_12:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,11);
              
               refPower.reload(StatsTestsQ31::POWERVALS_Q63_ID,mgr);
               
               outputPower.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ31::TEST_RMS_Q31_13:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,3);
              
               ref.reload(StatsTestsQ31::RMSVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ31::TEST_RMS_Q31_14:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,8);
              
               ref.reload(StatsTestsQ31::RMSVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ31::TEST_RMS_Q31_15:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,11);
              
               ref.reload(StatsTestsQ31::RMSVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ31::TEST_STD_Q31_16:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,3);
              
               ref.reload(StatsTestsQ31::STDVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ31::TEST_STD_Q31_17:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,8);
              
               ref.reload(StatsTestsQ31::STDVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ31::TEST_STD_Q31_18:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,11);
              
               ref.reload(StatsTestsQ31::STDVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ31::TEST_VAR_Q31_19:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,3);
              
               ref.reload(StatsTestsQ31::VARVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ31::TEST_VAR_Q31_20:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,8);
              
               ref.reload(StatsTestsQ31::VARVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ31::TEST_VAR_Q31_21:
            {
               inputA.reload(StatsTestsQ31::INPUT1_Q31_ID,mgr,11);
              
               ref.reload(StatsTestsQ31::VARVALS_Q31_ID,mgr);
               
               output.create(1,StatsTestsQ31::OUT_Q31_ID,mgr);

               refOffset = 2;
            }
            break;

          
        }
        
    }

    void StatsTestsQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      switch(id)
      {
            case StatsTestsQ31::TEST_MAX_Q31_1:
            case StatsTestsQ31::TEST_MAX_Q31_2:
            case StatsTestsQ31::TEST_MAX_Q31_3:
              index.dump(mgr);
              output.dump(mgr);
            break;

            case TEST_POWER_Q31_10:
            case TEST_POWER_Q31_11:
            case TEST_POWER_Q31_12:
              outputPower.dump(mgr);
            break;

            default:
              output.dump(mgr);
      }
    }
