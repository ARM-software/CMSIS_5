#include "FIRQ31.h"
#include "Error.h"

   
    void FIRQ31::test_fir_q31()
    {
       
       const q31_t *pSrc=samples.ptr();
       const q31_t *pCoefs=coefs.ptr();
       q31_t *pDst=output.ptr();


       arm_fir_q31(&instFir, pSrc, pDst, this->nbSamples);
        
    } 

    void FIRQ31::test_lms_q31()
    {
       
      const q31_t *pSrc=samples.ptr();
      const q31_t *pRef=refs.ptr();
      
      q31_t *pDst=output.ptr();
      q31_t *pErr=error.ptr();

      arm_lms_q31(&instLms, pSrc, (q31_t*)pRef, pDst, pErr,this->nbSamples);
        
    } 

    void FIRQ31::test_lms_norm_q31()
    {
      const q31_t *pSrc=samples.ptr();
      const q31_t *pRef=refs.ptr();
      
      q31_t *pDst=output.ptr();
      q31_t *pErr=error.ptr();

      arm_lms_norm_q31(&instLmsNorm, pSrc, (q31_t*)pRef, pDst, pErr,this->nbSamples);
        
    } 

   
    
    void FIRQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it;

       samples.reload(FIRQ31::SAMPLES1_Q31_ID,mgr,this->nbSamples);
       coefs.reload(FIRQ31::COEFS1_Q31_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,FIRQ31::STATE_Q31_ID,mgr);
       output.create(this->nbSamples,FIRQ31::OUT_SAMPLES_Q31_ID,mgr);

       switch(id)
       {
           case TEST_FIR_Q31_1:
              arm_fir_init_q31(&instFir,this->nbTaps,coefs.ptr(),state.ptr(),this->nbSamples);
           break;

           case TEST_LMS_Q31_2:
              refs.reload(FIRQ31::REFS1_Q31_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRQ31::ERR_Q31_ID,mgr);
              // Value of mu and postShift are arbitrary just for benchmark
              arm_lms_init_q31(&instLms,this->nbTaps,coefs.ptr(),state.ptr(),100,this->nbSamples,1);
           break;

           case TEST_LMS_NORM_Q31_3:
              refs.reload(FIRQ31::REFS1_Q31_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRQ31::ERR_Q31_ID,mgr);
              // Value of mu and postShift are arbitrary just for benchmark
              arm_lms_norm_init_q31(&instLmsNorm,this->nbTaps,coefs.ptr(),state.ptr(),100,this->nbSamples,1);
           break;
       }
       
    }

    void FIRQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
