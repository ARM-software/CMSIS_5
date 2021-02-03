#include "StatsTestsF64.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 300
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-14)

    void StatsTestsF64::test_entropy_f64()
    {
      const float64_t *inp  = inputA.ptr();
      const int16_t *dimsp  = dims.ptr();

      float64_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_entropy_f64(inp,dimsp[i+1]);
         outp++;
         inp += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);

    } 

    void StatsTestsF64::test_kullback_leibler_f64()
    {
      const float64_t *inpA  = inputA.ptr();
      const float64_t *inpB  = inputB.ptr();
      const int16_t *dimsp  = dims.ptr();

      float64_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_kullback_leibler_f64(inpA,inpB,dimsp[i+1]);
         outp++;
         inpA += dimsp[i+1];
         inpB += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 
  
  
    void StatsTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case StatsTestsF64::TEST_ENTROPY_F64_1:
            {
               inputA.reload(StatsTestsF64::INPUT22_F64_ID,mgr);
               dims.reload(StatsTestsF64::DIM22_S16_ID,mgr);
               ref.reload(StatsTestsF64::REF22_ENTROPY_F64_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF64::OUT_F64_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF64::TEST_KULLBACK_LEIBLER_F64_2:
            {
               inputA.reload(StatsTestsF64::INPUTA24_F64_ID,mgr);
               inputB.reload(StatsTestsF64::INPUTB24_F64_ID,mgr);
               dims.reload(StatsTestsF64::DIM24_S16_ID,mgr);
               ref.reload(StatsTestsF64::REF24_KL_F64_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF64::OUT_F64_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;
        }
        
    }

    void StatsTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
