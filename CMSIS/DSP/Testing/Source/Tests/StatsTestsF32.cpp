#include "StatsTestsF32.h"
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#include <cstdio>

#define SNR_THRESHOLD 120
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)

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

    void StatsTestsF32::test_entropy_f32()
    {
      const float32_t *inp  = inputA.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_entropy_f32(inp,this->vecDim);
         outp++;
         inp += vecDim;
      }

      ASSERT_NEAR_EQ(ref,output,(float32_t)1e-6);
    } 

    void StatsTestsF32::test_logsumexp_f32()
    {
      const float32_t *inp  = inputA.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_f32(inp,this->vecDim);
         outp++;
         inp += vecDim;
      }

      ASSERT_NEAR_EQ(ref,output,(float32_t)1e-6);
    } 


    void StatsTestsF32::test_kullback_leibler_f32()
    {
      const float32_t *inpA  = inputA.ptr();
      const float32_t *inpB  = inputB.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_kullback_leibler_f32(inpA,inpB,this->vecDim);
         outp++;
         inpA += vecDim;
         inpB += vecDim;
      }

      ASSERT_NEAR_EQ(ref,output,(float32_t)1e-6);
    } 

    void StatsTestsF32::test_logsumexp_dot_prod_f32()
    {
      const float32_t *inpA  = inputA.ptr();
      const float32_t *inpB  = inputB.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();
      float32_t *tmpp         = tmp.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_dot_prod_f32(inpA,inpB,this->vecDim,tmpp);
         outp++;
         inpA += vecDim;
         inpB += vecDim;
      }

      ASSERT_NEAR_EQ(ref,output,(float32_t)1e-6);
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
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,9);
              
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
               inputA.reload(StatsTestsF32::INPUT2_F32_ID,mgr,9);
              
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
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,9);
              
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
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,9);
              
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
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,9);
              
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
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,9);
              
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
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr,9);
              
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
               this->vecDim=dimsp[1];
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
               this->vecDim=dimsp[1];
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
               this->vecDim=dimsp[1];
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
               this->vecDim=dimsp[1];

               tmp.create(this->vecDim,StatsTestsF32::TMP_F32_ID,mgr);
            }
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
