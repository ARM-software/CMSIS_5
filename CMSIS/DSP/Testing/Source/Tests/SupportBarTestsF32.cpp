#include "SupportBarTestsF32.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "Test.h"


    void SupportBarTestsF32::test_barycenter_f32()
    {
       const float32_t *inp = input.ptr();
       const float32_t *coefsp = coefs.ptr();
       const int16_t *dimsp=dims.ptr();
       int nbVecs;
       int vecDim;

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbTests ; i ++)
       {
          nbVecs = dimsp[2*i+1];
          vecDim = dimsp[2*i+2];

        arm_barycenter_f32(inp, coefsp,
            outp, 
            nbVecs, 
            vecDim);
         
          inp += vecDim * nbVecs;
          coefsp += nbVecs;
          outp += vecDim;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
        ASSERT_EMPTY_TAIL(output);
    } 

  
    void SupportBarTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        dims.reload(SupportBarTestsF32::DIM_S16_ID,mgr);

        const int16_t *dimsp=dims.ptr();

        this->nbTests=dimsp[0];
       

        switch(id)
        {
           
            case TEST_BARYCENTER_F32_1:
              input.reload(SupportBarTestsF32::SAMPLES_F32_ID,mgr);
              coefs.reload(SupportBarTestsF32::COEFS_F32_ID,mgr);
              ref.reload(SupportBarTestsF32::REF_F32_ID,mgr);

              output.create(ref.nbSamples(),SupportBarTestsF32::OUT_SAMPLES_F32_ID,mgr);
            break;
        }

       

    }

    void SupportBarTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
