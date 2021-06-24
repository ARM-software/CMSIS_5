#include "SupportBarTestsF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


    void SupportBarTestsF16::test_barycenter_f16()
    {
       const float16_t *inp = input.ptr();
       const float16_t *coefsp = coefs.ptr();
       const int16_t *dimsp=dims.ptr();
       int nbVecs;
       int vecDim;

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbTests ; i ++)
       {
          nbVecs = dimsp[2*i+1];
          vecDim = dimsp[2*i+2];

        arm_barycenter_f16(inp, coefsp,
            outp, 
            nbVecs, 
            vecDim);
         
          inp += vecDim * nbVecs;
          coefsp += nbVecs;
          outp += vecDim;
       }

        ASSERT_NEAR_EQ(output,ref,(float16_t)1e-3);
        ASSERT_EMPTY_TAIL(output);
    } 

  
    void SupportBarTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        dims.reload(SupportBarTestsF16::DIM_S16_ID,mgr);

        const int16_t *dimsp=dims.ptr();

        this->nbTests=dimsp[0];
       

        switch(id)
        {
           
            case TEST_BARYCENTER_F16_1:
              input.reload(SupportBarTestsF16::SAMPLES_F16_ID,mgr);
              coefs.reload(SupportBarTestsF16::COEFS_F16_ID,mgr);
              ref.reload(SupportBarTestsF16::REF_F16_ID,mgr);

              output.create(ref.nbSamples(),SupportBarTestsF16::OUT_SAMPLES_F16_ID,mgr);
            break;
        }

       

    }

    void SupportBarTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
