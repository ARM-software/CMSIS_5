#include "BIQUADF64.h"
#include "Error.h"

   
   
    void BIQUADF64::test_biquad_cascade_df2T_f64()
    {
       arm_biquad_cascade_df2T_f64(&instBiquadDf2T, (float64_t *)this->pSrc, this->pDst, this->nbSamples);
    } 



    
    void BIQUADF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       
       std::vector<Testing::param_t>::iterator it = params.begin();
       this->numStages = *it++;
       this->nbSamples = *it;

       
       

       switch(id)
       {
          
           case TEST_BIQUAD_CASCADE_DF2T_F64_1:
                  samples.reload(BIQUADF64::SAMPLES1_F64_ID,mgr,this->nbSamples);
                  output.create(this->nbSamples,BIQUADF64::OUT_SAMPLES_F64_ID,mgr);
                  coefs.reload(BIQUADF64::COEFS1_F64_ID,mgr,this->numStages * 5);
                  state.create(2*this->numStages,BIQUADF64::STATE_F64_ID,mgr);

                  arm_biquad_cascade_df2T_init_f64(&instBiquadDf2T,
                    this->numStages,
                    coefs.ptr(),
                    state.ptr());
           break;

          
       }

       this->pSrc=samples.ptr();
       this->pDst=output.ptr();
       
    }

    void BIQUADF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
