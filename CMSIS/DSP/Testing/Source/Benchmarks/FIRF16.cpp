#include "FIRF16.h"
#include "Error.h"

   
    void FIRF16::test_fir_f16()
    {
       arm_fir_f16(&instFir, this->pSrc, this->pDst, this->nbSamples);
    } 


   
    
    void FIRF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it;

       samples.reload(FIRF16::SAMPLES1_F16_ID,mgr,this->nbSamples);
       coefs.reload(FIRF16::COEFS1_F16_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,FIRF16::STATE_F16_ID,mgr);
       output.create(this->nbSamples,FIRF16::OUT_SAMPLES_F16_ID,mgr);

       switch(id)
       {
           case TEST_FIR_F16_1:
              arm_fir_init_f16(&instFir,this->nbTaps,coefs.ptr(),state.ptr(),this->nbSamples);

              this->pSrc=samples.ptr();
              this->pCoefs=coefs.ptr();
              this->pDst=output.ptr();
           break;

          

           
       }
       
    }

    void FIRF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
