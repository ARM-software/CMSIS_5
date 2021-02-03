#include "FIRQ15.h"
#include "Error.h"

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
static __ALIGNED(8) q15_t coeffArray[64];
#endif 
   
    void FIRQ15::test_fir_q15()
    {
       arm_fir_q15(&instFir, this->pSrc, this->pDst, this->nbSamples);
    } 

    void FIRQ15::test_lms_q15()
    {
      arm_lms_q15(&instLms, this->pSrc, (q15_t*)this->pRef, this->pDst, this->pErr,this->nbSamples);
    } 

    void FIRQ15::test_lms_norm_q15()
    {
      arm_lms_norm_q15(&instLmsNorm, this->pSrc, (q15_t*)this->pRef, this->pDst, this->pErr,this->nbSamples); 
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
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
              /* Copy coefficients and pad to zero 
              */
              memset(coeffArray,0,32*sizeof(q15_t));
              q15_t *ptr;

              ptr=coefs.ptr();
              memcpy(coeffArray,ptr,this->nbTaps*sizeof(q15_t));
              this->pCoefs = coeffArray;
#else
              this->pCoefs=coefs.ptr();
#endif
              arm_fir_init_q15(&instFir,this->nbTaps,coefs.ptr(),state.ptr(),this->nbSamples);

              this->pSrc=samples.ptr();
              this->pDst=output.ptr();
           break;

           case TEST_LMS_Q15_2:
              refs.reload(FIRQ15::REFS1_Q15_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRQ15::ERR_Q15_ID,mgr);
              arm_lms_init_q15(&instLms,this->nbTaps,coefs.ptr(),state.ptr(),100,this->nbSamples,1);

              this->pSrc=samples.ptr();
              this->pRef=refs.ptr();
      
              this->pDst=output.ptr();
              this->pErr=error.ptr();
           break;

           case TEST_LMS_NORM_Q15_3:
              refs.reload(FIRQ15::REFS1_Q15_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRQ15::ERR_Q15_ID,mgr);
              arm_lms_norm_init_q15(&instLmsNorm,this->nbTaps,coefs.ptr(),state.ptr(),100,this->nbSamples,1);

              this->pSrc=samples.ptr();
              this->pRef=refs.ptr();
      
              this->pDst=output.ptr();
              this->pErr=error.ptr();
           break;
       }
       
    }

    void FIRQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
