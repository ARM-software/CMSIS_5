#include "SupportTestsF32.h"
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#include <cstdio>


    void SupportTestsF32::test_weighted_sum_f32()
    {
       const float32_t *inp = input.ptr();
       const float32_t *coefsp = coefs.ptr();

       float32_t *outp = output.ptr();
       
      
       *outp=arm_weighted_sum_f32(inp, coefsp,input.nbSamples());
         
          
        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

  
    void SupportTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        switch(id)
        {
           
            case TEST_WEIGHTED_SUM_F32_1:
              input.reload(SupportTestsF32::INPUTS_F32_ID,mgr);
              coefs.reload(SupportTestsF32::WEIGHTS_F32_ID,mgr);
              ref.reload(SupportTestsF32::REF_F32_ID,mgr);

              output.create(ref.nbSamples(),SupportTestsF32::OUT_F32_ID,mgr);
            break;
        }

       

    }

    void SupportTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
