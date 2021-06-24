#include "StatsTestsQ15.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

//#include <cstdio>

#define SNR_THRESHOLD 50
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)100)
#define ABS_ERROR_Q63 (1<<17)

    void StatsTestsQ15::test_max_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;
        uint32_t  indexval;

        q15_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        q15_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_max_q15(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsQ15::test_absmax_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;
        uint32_t  indexval;

        q15_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        q15_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_absmax_q15(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsQ15::test_min_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;
        uint32_t  indexval;

        q15_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        q15_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_min_q15(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsQ15::test_absmin_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;
        uint32_t  indexval;

        q15_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        q15_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_absmin_q15(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsQ15::test_mean_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;

        q15_t *refp  = ref.ptr();

        q15_t *outp  = output.ptr();

        arm_mean_q15(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q15);

    }

    void StatsTestsQ15::test_power_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q63_t result;

        q63_t *refp  = refPower.ptr();

        q63_t *outp  = outputPower.ptr();

        arm_power_q15(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],(q63_t)ABS_ERROR_Q63);

    }

    void StatsTestsQ15::test_rms_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;

        q15_t *refp  = ref.ptr();

        q15_t *outp  = output.ptr();

        arm_rms_q15(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q15);

    }

    void StatsTestsQ15::test_std_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;

        q15_t *refp  = ref.ptr();

        q15_t *outp  = output.ptr();

        arm_std_q15(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q15);

    }

    void StatsTestsQ15::test_var_q15()
    {
        const q15_t *inp  = inputA.ptr();

        q15_t result;

        q15_t *refp  = ref.ptr();

        q15_t *outp  = output.ptr();

        arm_var_q15(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(result,refp[this->refOffset],ABS_ERROR_Q15);

    }

  
  
    void StatsTestsQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case StatsTestsQ15::TEST_MAX_Q15_1:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,7);
              
               maxIndexes.reload(StatsTestsQ15::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::MAXVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_MAX_Q15_2:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,16);
              
               maxIndexes.reload(StatsTestsQ15::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::MAXVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_MAX_Q15_3:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,23);
              
               maxIndexes.reload(StatsTestsQ15::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::MAXVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_MEAN_Q15_4:
            {
               inputA.reload(StatsTestsQ15::INPUT2_Q15_ID,mgr,7);
              
               ref.reload(StatsTestsQ15::MEANVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_MEAN_Q15_5:
            {
               inputA.reload(StatsTestsQ15::INPUT2_Q15_ID,mgr,16);
              
               ref.reload(StatsTestsQ15::MEANVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_MEAN_Q15_6:
            {
               inputA.reload(StatsTestsQ15::INPUT2_Q15_ID,mgr,23);
              
               ref.reload(StatsTestsQ15::MEANVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_MIN_Q15_7:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,7);
              
               minIndexes.reload(StatsTestsQ15::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::MINVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_MIN_Q15_8:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,16);
              
               minIndexes.reload(StatsTestsQ15::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::MINVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_MIN_Q15_9:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,23);
              
               minIndexes.reload(StatsTestsQ15::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::MINVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_POWER_Q15_10:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,7);
              
               refPower.reload(StatsTestsQ15::POWERVALS_Q63_ID,mgr);
               
               outputPower.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_POWER_Q15_11:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,16);
              
               refPower.reload(StatsTestsQ15::POWERVALS_Q63_ID,mgr);
               
               outputPower.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_POWER_Q15_12:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,23);
              
               refPower.reload(StatsTestsQ15::POWERVALS_Q63_ID,mgr);
               
               outputPower.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_RMS_Q15_13:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,7);
              
               ref.reload(StatsTestsQ15::RMSVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_RMS_Q15_14:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,16);
              
               ref.reload(StatsTestsQ15::RMSVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_RMS_Q15_15:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,23);
              
               ref.reload(StatsTestsQ15::RMSVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_STD_Q15_16:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,7);
              
               ref.reload(StatsTestsQ15::STDVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_STD_Q15_17:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,16);
              
               ref.reload(StatsTestsQ15::STDVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_STD_Q15_18:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,23);
              
               ref.reload(StatsTestsQ15::STDVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_VAR_Q15_19:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,7);
              
               ref.reload(StatsTestsQ15::VARVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_VAR_Q15_20:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,16);
              
               ref.reload(StatsTestsQ15::VARVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_VAR_Q15_21:
            {
               inputA.reload(StatsTestsQ15::INPUT1_Q15_ID,mgr,23);
              
               ref.reload(StatsTestsQ15::VARVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_ABSMAX_Q15_22:
            {
               inputA.reload(StatsTestsQ15::INPUTNEW1_Q15_ID,mgr,7);
              
               maxIndexes.reload(StatsTestsQ15::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::ABSMAXVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_ABSMAX_Q15_23:
            {
               inputA.reload(StatsTestsQ15::INPUTNEW1_Q15_ID,mgr,16);
              
               maxIndexes.reload(StatsTestsQ15::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::ABSMAXVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_ABSMAX_Q15_24:
            {
               inputA.reload(StatsTestsQ15::INPUTNEW1_Q15_ID,mgr,23);
              
               maxIndexes.reload(StatsTestsQ15::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::ABSMAXVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsQ15::TEST_ABSMIN_Q15_25:
            {
               inputA.reload(StatsTestsQ15::INPUTNEW1_Q15_ID,mgr,7);
              
               minIndexes.reload(StatsTestsQ15::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::ABSMINVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsQ15::TEST_ABSMIN_Q15_26:
            {
               inputA.reload(StatsTestsQ15::INPUTNEW1_Q15_ID,mgr,16);
              
               minIndexes.reload(StatsTestsQ15::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::ABSMINVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsQ15::TEST_ABSMIN_Q15_27:
            {
               inputA.reload(StatsTestsQ15::INPUTNEW1_Q15_ID,mgr,23);
              
               minIndexes.reload(StatsTestsQ15::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsQ15::ABSMINVALS_Q15_ID,mgr);
               
               output.create(1,StatsTestsQ15::OUT_Q15_ID,mgr);
               index.create(1,StatsTestsQ15::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

          
        }
        
    }

    void StatsTestsQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      switch(id)
      {
            case StatsTestsQ15::TEST_MAX_Q15_1:
            case StatsTestsQ15::TEST_MAX_Q15_2:
            case StatsTestsQ15::TEST_MAX_Q15_3:
            case StatsTestsQ15::TEST_MIN_Q15_7:
            case StatsTestsQ15::TEST_MIN_Q15_8:
            case StatsTestsQ15::TEST_MIN_Q15_9:
              index.dump(mgr);
              output.dump(mgr);
            break;

            case TEST_POWER_Q15_10:
            case TEST_POWER_Q15_11:
            case TEST_POWER_Q15_12:
              outputPower.dump(mgr);
            break;

            default:
              output.dump(mgr);
      }
    }
