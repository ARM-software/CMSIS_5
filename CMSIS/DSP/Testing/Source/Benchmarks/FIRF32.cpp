#include "FIRF32.h"
#include "Error.h"

   
    void FIRF32::test_fir_f32()
    {
       
       const float32_t *pSrc=samples.ptr();
       const float32_t *pCoefs=coefs.ptr();
       float32_t *pDst=output.ptr();


       arm_fir_f32(&instFir, pSrc, pDst, this->nbSamples);
        
    } 

    void FIRF32::test_lms_f32()
    {
       
      const float32_t *pSrc=samples.ptr();
      const float32_t *pRef=refs.ptr();
      
      float32_t *pDst=output.ptr();
      float32_t *pErr=error.ptr();

      arm_lms_f32(&instLms, pSrc, (float32_t*)pRef, pDst, pErr,this->nbSamples);
        
    } 

    void FIRF32::test_lms_norm_f32()
    {
      const float32_t *pSrc=samples.ptr();
      const float32_t *pRef=refs.ptr();
      
      float32_t *pDst=output.ptr();
      float32_t *pErr=error.ptr();

      arm_lms_norm_f32(&instLmsNorm, pSrc, (float32_t*)pRef, pDst, pErr,this->nbSamples);
        
    } 

   
    
    void FIRF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it;

       samples.reload(FIRF32::SAMPLES1_F32_ID,mgr,this->nbSamples);
       coefs.reload(FIRF32::COEFS1_F32_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,FIRF32::STATE_F32_ID,mgr);
       output.create(this->nbSamples,FIRF32::OUT_SAMPLES_F32_ID,mgr);

       switch(id)
       {
           case TEST_FIR_F32_1:
              arm_fir_init_f32(&instFir,this->nbTaps,coefs.ptr(),state.ptr(),this->nbSamples);
           break;

           case TEST_LMS_F32_2:
              refs.reload(FIRF32::REFS1_F32_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRF32::ERR_F32_ID,mgr);
              arm_lms_init_f32(&instLms,this->nbTaps,coefs.ptr(),state.ptr(),0.1,this->nbSamples);
           break;

           case TEST_LMS_NORM_F32_3:
              refs.reload(FIRF32::REFS1_F32_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRF32::ERR_F32_ID,mgr);
              arm_lms_norm_init_f32(&instLmsNorm,this->nbTaps,coefs.ptr(),state.ptr(),0.1,this->nbSamples);
           break;
       }
       
    }

    void FIRF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
