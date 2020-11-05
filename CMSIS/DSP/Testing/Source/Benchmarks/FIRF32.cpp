#include "FIRF32.h"
#include "Error.h"

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
static __ALIGNED(8) float32_t coeffArray[64];
#endif 

    void FIRF32::test_fir_f32()
    {
       arm_fir_f32(&instFir, this->pSrc, this->pDst, this->nbSamples);
    } 

    void FIRF32::test_lms_f32()
    {
      arm_lms_f32(&instLms, this->pSrc, (float32_t*)this->pRef, this->pDst, this->pErr,this->nbSamples); 
    } 

    void FIRF32::test_lms_norm_f32()
    {
       arm_lms_norm_f32(&instLmsNorm, this->pSrc, (float32_t*)this->pRef, this->pDst, this->pErr,this->nbSamples);
    } 

   
    
    void FIRF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it;

       samples.reload(FIRF32::SAMPLES1_F32_ID,mgr,this->nbSamples);
       coefs.reload(FIRF32::COEFS1_F32_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbSamples + this->nbTaps - 1,FIRF32::STATE_F32_ID,mgr);
       output.create(this->nbSamples,FIRF32::OUT_SAMPLES_F32_ID,mgr);

       switch(id)
       {
           case TEST_FIR_F32_1:

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
              /* Copy coefficients and pad to zero 
              */
              memset(coeffArray,0,32*sizeof(float32_t));
              float32_t *ptr;

              ptr=coefs.ptr();
              memcpy(coeffArray,ptr,this->nbTaps*sizeof(float32_t));
              this->pCoefs = coeffArray;
#else
              this->pCoefs=coefs.ptr();
#endif

              this->pSrc=samples.ptr();
              
              this->pDst=output.ptr();

              arm_fir_init_f32(&instFir,this->nbTaps,this->pCoefs,state.ptr(),this->nbSamples);
           break;

           case TEST_LMS_F32_2:
              refs.reload(FIRF32::REFS1_F32_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRF32::ERR_F32_ID,mgr);
              arm_lms_init_f32(&instLms,this->nbTaps,coefs.ptr(),state.ptr(),0.1,this->nbSamples);

              this->pSrc=samples.ptr();
              this->pRef=refs.ptr();
      
              this->pDst=output.ptr();
              this->pErr=error.ptr();
           break;

           case TEST_LMS_NORM_F32_3:
              refs.reload(FIRF32::REFS1_F32_ID,mgr,this->nbSamples);
              error.create(this->nbSamples,FIRF32::ERR_F32_ID,mgr);
              arm_lms_norm_init_f32(&instLmsNorm,this->nbTaps,coefs.ptr(),state.ptr(),0.1,this->nbSamples);

              this->pSrc=samples.ptr();
              this->pRef=refs.ptr();
      
              this->pDst=output.ptr();
              this->pErr=error.ptr();
           break;
       }
       
    }

    void FIRF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
