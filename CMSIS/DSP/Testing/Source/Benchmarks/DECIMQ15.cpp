#include "DECIMQ15.h"
#include "Error.h"

   
    void DECIMQ15::test_fir_decimate_q15()
    {
       arm_fir_decimate_q15(&instDecim,this->pSrc,this->pDst,this->nbSamples);
    } 

 
   
   
    void DECIMQ15::test_fir_interpolate_q15()
    {
       arm_fir_interpolate_q15(&instInterpol,this->pSrc,this->pDst,this->nbSamples);
    } 
    
    void DECIMQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it++;
       

       samples.reload(DECIMQ15::SAMPLES1_Q15_ID,mgr,this->nbSamples);
       coefs.reload(DECIMQ15::COEFS1_Q15_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,DECIMQ15::STATE_Q15_ID,mgr);
       output.create(this->nbSamples,DECIMQ15::OUT_SAMPLES_Q15_ID,mgr);

       switch(id)
       {
           case TEST_FIR_DECIMATE_Q15_1:
              this->decimationFactor = *it;
              arm_fir_decimate_init_q15(&instDecim,
                 this->nbTaps,
                 this->decimationFactor,
                 coefs.ptr(),
                 state.ptr(),
                 this->nbSamples);
           break;


           case TEST_FIR_INTERPOLATE_Q15_2:
              this->interpolationFactor = *it;
              arm_fir_interpolate_init_q15(&instInterpol,
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

    void DECIMQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
