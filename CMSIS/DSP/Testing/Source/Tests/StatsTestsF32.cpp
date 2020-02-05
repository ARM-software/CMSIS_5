#include "StatsTestsF32.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "Test.h"


#define SNR_THRESHOLD 120
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-5)

    void StatsTestsF32::test_max_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;
        uint32_t  indexval;

        float32_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        float32_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_max_f32(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF32::test_max_no_idx_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;

        float32_t *refp  = ref.ptr();

        float32_t *outp  = output.ptr();

        arm_max_no_idx_f32(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_EQ(result,refp[this->refOffset]);

    }

    void StatsTestsF32::test_min_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;
        uint32_t  indexval;

        float32_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        float32_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_min_f32(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF32::test_mean_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;

        float32_t *refp  = ref.ptr();

        float32_t *outp  = output.ptr();

        arm_mean_f32(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF32::test_power_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;

        float32_t *refp  = ref.ptr();

        float32_t *outp  = output.ptr();

        arm_power_f32(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF32::test_rms_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;

        float32_t *refp  = ref.ptr();

        float32_t *outp  = output.ptr();

        arm_rms_f32(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF32::test_std_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;

        float32_t *refp  = ref.ptr();

        float32_t *outp  = output.ptr();

        arm_std_f32(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF32::test_var_f32()
    {
        const float32_t *inp  = inputA.ptr();

        float32_t result;

        float32_t *refp  = ref.ptr();

        float32_t *outp  = output.ptr();

        arm_var_f32(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }


    void StatsTestsF32::test_std_stability_f32()
    {
      /*

      With the textbook algorithm, those values will produce a negative
      value for the variance.

      The CMSIS-DSP variance algorithm is the two pass one so will work
      with those values.

      So, it should be possible to compute the square root for the standard
      deviation.

      */
      float32_t in[4]={4.0f, 7.0f, 13.0f, 16.0f};
      float32_t result;
      int i;

      /*

      Add bigger offset so that average is much bigger than standard deviation.

      */
      for(i=0 ; i < 4; i++)
      {
        in[i] += 3.0e4f;
      }

      arm_std_f32(in,4,&result);

      /*

      If variance is giving a negative value, the square root
      should return zero.

      We check it is not happening here.


      */

      ASSERT_TRUE(fabs(5.47723f - result) < 1.0e-4f);

    }

    void StatsTestsF32::test_entropy_f32()
    {
      const float32_t *inp  = inputA.ptr();
      const int16_t *dimsp  = dims.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_entropy_f32(inp,dimsp[i+1]);
         outp++;
         inp += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);

    } 

    void StatsTestsF32::test_logsumexp_f32()
    {
      const float32_t *inp  = inputA.ptr();
      const int16_t *dimsp  = dims.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_f32(inp,dimsp[i+1]);
         outp++;
         inp += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 


    void StatsTestsF32::test_kullback_leibler_f32()
    {
      const float32_t *inpA  = inputA.ptr();
      const float32_t *inpB  = inputB.ptr();
      const int16_t *dimsp  = dims.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_kullback_leibler_f32(inpA,inpB,dimsp[i+1]);
         outp++;
         inpA += dimsp[i+1];
         inpB += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 

    void StatsTestsF32::test_logsumexp_dot_prod_f32()
    {
      const float32_t *inpA  = inputA.ptr();
      const float32_t *inpB  = inputB.ptr();
      const int16_t *dimsp  = dims.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();
      float32_t *tmpp         = tmp.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_dot_prod_f32(inpA,inpB,dimsp[i+1],tmpp);
         outp++;
         inpA += dimsp[i+1];
         inpB += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 

   
  
    void StatsTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        switch(id)
        {
            case StatsTestsF32::TEST_MAX_F32_1:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,3);
              
               maxIndexes.reload(StatsTestsF32::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF32::MAXVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);
               index.create(1,StatsTestsF32::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_MAX_F32_2:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,8);
              
               maxIndexes.reload(StatsTestsF32::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF32::MAXVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);
               index.create(1,StatsTestsF32::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_MAX_F32_3:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,11);
              
               maxIndexes.reload(StatsTestsF32::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF32::MAXVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);
               index.create(1,StatsTestsF32::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF32::TEST_MEAN_F32_4:
            {
               inputA.reload(StatsTestsF32::INPUT2_F32_ID,mgr,3);
              
               ref.reload(StatsTestsF32::MEANVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_MEAN_F32_5:
            {
               inputA.reload(StatsTestsF32::INPUT2_F32_ID,mgr,8);
              
               ref.reload(StatsTestsF32::MEANVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_MEAN_F32_6:
            {
               inputA.reload(StatsTestsF32::INPUT2_F32_ID,mgr,11);
              
               ref.reload(StatsTestsF32::MEANVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF32::TEST_MIN_F32_7:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,3);
              
               minIndexes.reload(StatsTestsF32::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF32::MINVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);
               index.create(1,StatsTestsF32::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_MIN_F32_8:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,8);
              
               minIndexes.reload(StatsTestsF32::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF32::MINVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);
               index.create(1,StatsTestsF32::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_MIN_F32_9:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,11);
              
               minIndexes.reload(StatsTestsF32::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF32::MINVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);
               index.create(1,StatsTestsF32::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF32::TEST_POWER_F32_10:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,3);
              
               ref.reload(StatsTestsF32::POWERVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_POWER_F32_11:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,8);
              
               ref.reload(StatsTestsF32::POWERVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_POWER_F32_12:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,11);
              
               ref.reload(StatsTestsF32::POWERVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF32::TEST_RMS_F32_13:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,3);
              
               ref.reload(StatsTestsF32::RMSVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_RMS_F32_14:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,8);
              
               ref.reload(StatsTestsF32::RMSVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_RMS_F32_15:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,11);
              
               ref.reload(StatsTestsF32::RMSVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF32::TEST_STD_F32_16:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,3);
              
               ref.reload(StatsTestsF32::STDVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_STD_F32_17:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,8);
              
               ref.reload(StatsTestsF32::STDVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_STD_F32_18:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,11);
              
               ref.reload(StatsTestsF32::STDVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF32::TEST_VAR_F32_19:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,3);
              
               ref.reload(StatsTestsF32::VARVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_VAR_F32_20:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,8);
              
               ref.reload(StatsTestsF32::VARVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_VAR_F32_21:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,11);
              
               ref.reload(StatsTestsF32::VARVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF32::TEST_ENTROPY_F32_22:
            {
               inputA.reload(StatsTestsF32::INPUT22_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM22_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF22_ENTROPY_F32_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF32::OUT_F32_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF32::TEST_LOGSUMEXP_F32_23:
            {
               inputA.reload(StatsTestsF32::INPUT23_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM23_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF23_LOGSUMEXP_F32_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF32::OUT_F32_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF32::TEST_KULLBACK_LEIBLER_F32_24:
            {
               inputA.reload(StatsTestsF32::INPUTA24_F32_ID,mgr);
               inputB.reload(StatsTestsF32::INPUTB24_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM24_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF24_KL_F32_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF32::OUT_F32_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF32::TEST_LOGSUMEXP_DOT_PROD_F32_25:
            {
               inputA.reload(StatsTestsF32::INPUTA25_F32_ID,mgr);
               inputB.reload(StatsTestsF32::INPUTB25_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM25_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF25_LOGSUMEXP_DOT_F32_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF32::OUT_F32_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];

               /* 12 is max vecDim as defined in Python script generating the data */
               tmp.create(12,StatsTestsF32::TMP_F32_ID,mgr);
            }
            break;

            case StatsTestsF32::TEST_MAX_NO_IDX_F32_26:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,3);
              
               ref.reload(StatsTestsF32::MAXVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF32::TEST_MAX_NO_IDX_F32_27:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,8);
              
               ref.reload(StatsTestsF32::MAXVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF32::TEST_MAX_NO_IDX_F32_28:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,11);
              
               ref.reload(StatsTestsF32::MAXVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 2;
            }
            break;

            case TEST_MEAN_F32_29:
               inputA.reload(StatsTestsF32::INPUT2_F32_ID,mgr,100);
              
               ref.reload(StatsTestsF32::MEANVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 3;
            break;

            case TEST_RMS_F32_30:
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,100);
              
               ref.reload(StatsTestsF32::RMSVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 3;
            break;

            case TEST_STD_F32_31:
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,100);
              
               ref.reload(StatsTestsF32::STDVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 3;
            break;

            case TEST_VAR_F32_32:
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,100);
              
               ref.reload(StatsTestsF32::VARVALS_F32_ID,mgr);
               
               output.create(1,StatsTestsF32::OUT_F32_ID,mgr);

               refOffset = 3;
            break;
        }
        
    }

    void StatsTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      switch(id)
      {
            case StatsTestsF32::TEST_MAX_F32_1:
            case StatsTestsF32::TEST_MAX_F32_2:
            case StatsTestsF32::TEST_MAX_F32_3:
            case StatsTestsF32::TEST_MIN_F32_7:
            case StatsTestsF32::TEST_MIN_F32_8:
            case StatsTestsF32::TEST_MIN_F32_9:
              index.dump(mgr);
              output.dump(mgr);
            break;

            default:
              output.dump(mgr);
      }
    }
