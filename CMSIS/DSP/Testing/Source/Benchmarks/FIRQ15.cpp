#include "FIRQ15.h"
#include "Error.h"

   
    void FIRQ15::test_fir_q15()
    {
       
       const q15_t *pSrc=samples.ptr();
       const q15_t *pCoefs=coefs.ptr();
       q15_t *pDst=output.ptr();


       arm_fir_q15(&instFir, pSrc, pDst, this->nbSamples);
        
    } 

    void FIRQ15::test_lms_q15()
    {
       
      const q15_t *pSrc=samples.ptr();
      const q15_t *pRef=refs.ptr();
      
      q15_t *pDst=output.ptr();
      q15_t *pErr=error.ptr();

      arm_lms_q15(&instLms, pSrc, (q15_t*)pRef, pDst, pErr,this->nbSamples);
        
    } 

    void FIRQ15::test_lms_norm_q15()
    {
      const q15_t *pSrc=samples.ptr();
      const q15_t *pRef=refs.ptr();
      
      q15_t *pDst=output.ptr();
      q15_t *pErr=error.ptr();

      arm_lms_norm_q15(&instLmsNorm, pSrc, (q15_t*)pRef, pDst, pErr,this->nbSamples);
        
    } 

   
    
    void FIRQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it;

       samples.reload(FIRQ15::SAMPLES1_Q15_ID,mgr,this->nbSamples);
       coefs.reload(FIRQ15::COEFS1_Q15_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,FIRQ15::STATE_Q15_ID,mgr);
       output.create(this->nbSamples,FIRQ15::OUT_SAMPLES_Q15_ID,mgr);

       switch(id)
       {
           case TEST_FIR_Q15_1:
              arm_fir_init_q15(&instFir,this->nbTaps,coefs.ptr(),state.ptr(),this->nbSamples);
           break;

           case TEST_LMS_Q15_2:
              refs.reload(FIRQ15::REFS1_Q15_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRQ15::ERR_Q15_ID,mgr);
              arm_lms_init_q15(&instLms,this->nbTaps,coefs.ptr(),state.ptr(),100,this->nbSamples,1);
           break;

           case TEST_LMS_NORM_Q15_3:
              refs.reload(FIRQ15::REFS1_Q15_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRQ15::ERR_Q15_ID,mgr);
              arm_lms_norm_init_q15(&instLmsNorm,this->nbTaps,coefs.ptr(),state.ptr(),100,this->nbSamples,1);
           break;
       }
       
    }

    void FIRQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
