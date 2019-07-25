#include "StatsTestsF32.h"
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#include <cstdio>


    void StatsTestsF32::test_entropy_f32()
    {
      const float32_t *inp  = inputA.ptr();

      float32_t *refp         = ref.ptr();
      float32_t *outp         = output.ptr();
      float32_t *result;

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
      float32_t *result;

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
      float32_t *result;

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
      float32_t *result;

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
            case StatsTestsF32::TEST_ENTROPY_F32_1:
            {
               inputA.reload(StatsTestsF32::INPUT1_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM1_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF1_ENTROPY_F32_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF32::OUT_F32_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
               this->vecDim=dimsp[1];
            }
            break;

            case StatsTestsF32::TEST_LOGSUMEXP_F32_2:
            {
               inputA.reload(StatsTestsF32::INPUT2_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM2_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF2_LOGSUMEXP_F32_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF32::OUT_F32_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
               this->vecDim=dimsp[1];
            }
            break;

            case StatsTestsF32::TEST_KULLBACK_LEIBLER_F32_3:
            {
               inputA.reload(StatsTestsF32::INPUTA3_F32_ID,mgr);
               inputB.reload(StatsTestsF32::INPUTB3_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM3_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF3_KL_F32_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF32::OUT_F32_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
               this->vecDim=dimsp[1];
            }
            break;

            case StatsTestsF32::TEST_LOGSUMEXP_DOT_PROD_F32_4:
            {
               inputA.reload(StatsTestsF32::INPUTA4_F32_ID,mgr);
               inputB.reload(StatsTestsF32::INPUTB4_F32_ID,mgr);
               dims.reload(StatsTestsF32::DIM4_S16_ID,mgr);
               ref.reload(StatsTestsF32::REF4_LOGSUMEXP_DOT_F32_ID,mgr);
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
       output.dump(mgr);
    }
