#include "DECIMQ31.h"
#include "Error.h"

   
    void DECIMQ31::test_fir_decimate_q31()
    {
       arm_fir_decimate_q31(&instDecim,this->pSrc,this->pDst,this->nbSamples);
    } 

 
   
   
    void DECIMQ31::test_fir_interpolate_q31()
    {
       arm_fir_interpolate_q31(&instInterpol,this->pSrc,this->pDst,this->nbSamples);
    } 
    
    void DECIMQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it++;
       

       samples.reload(DECIMQ31::SAMPLES1_Q31_ID,mgr,this->nbSamples);
       coefs.reload(DECIMQ31::COEFS1_Q31_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,DECIMQ31::STATE_Q31_ID,mgr);
       output.create(this->nbSamples,DECIMQ31::OUT_SAMPLES_Q31_ID,mgr);

       switch(id)
       {
           case TEST_FIR_DECIMATE_Q31_1:
              this->decimationFactor = *it;
              arm_fir_decimate_init_q31(&instDecim,
                 this->nbTaps,
                 this->decimationFactor,
                 coefs.ptr(),
                 state.ptr(),
                 this->nbSamples);
           break;


           case TEST_FIR_INTERPOLATE_Q31_2:
              this->interpolationFactor = *it;
              arm_fir_interpolate_init_q31(&instInterpol,
                 this->interpolationFactor,
                 this->nbTaps,
                 coefs.ptr(),
                 state.ptr(),
                 this->nbSamples);
           break;

          
       }

       this->pSrc=samples.ptr();
       this->pDst=output.ptr();
       
    }

    void DECIMQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
