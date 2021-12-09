#include "FIRF16.h"
#include "Error.h"

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
static __ALIGNED(8) float16_t coeffArray[64];
#endif 
   
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

       state.create(ROUND_UP(this->nbSamples,8) + this->nbSamples + this->nbTaps - 1,FIRF16::STATE_F16_ID,mgr);
       output.create(this->nbSamples,FIRF16::OUT_SAMPLES_F16_ID,mgr);

       switch(id)
       {
           case TEST_FIR_F16_1:
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
              /* Copy coefficients and pad to zero 
              */
              memset(coeffArray,0,32*sizeof(float16_t));
              float16_t *ptr;

              ptr=coefs.ptr();
              memcpy(coeffArray,ptr,this->nbTaps*sizeof(float16_t));
              this->pCoefs = coeffArray;
#else
              this->pCoefs=coefs.ptr();
#endif

              arm_fir_init_f16(&instFir,this->nbTaps,coefs.ptr(),state.ptr(),this->nbSamples);

              this->pSrc=samples.ptr();
              this->pDst=output.ptr();
           break;

          

           
       }
       
    }

    void FIRF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
    }
