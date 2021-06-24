#include "FIRQ7.h"
#include "Error.h"

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
static __ALIGNED(8) q7_t coeffArray[64];
#endif 
   
    void FIRQ7::test_fir_q7()
    {
       arm_fir_q7(&instFir, this->pSrc, this->pDst, this->nbSamples);
    } 

   

    
    void FIRQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it;

       samples.reload(FIRQ7::SAMPLES1_Q7_ID,mgr,this->nbSamples);
       coefs.reload(FIRQ7::COEFS1_Q7_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,FIRQ7::STATE_Q7_ID,mgr);
       output.create(this->nbSamples,FIRQ7::OUT_SAMPLES_Q7_ID,mgr);

       switch(id)
       {
           case TEST_FIR_Q7_1:
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
              /* Copy coefficients and pad to zero 
              */
              memset(coeffArray,0,32*sizeof(q7_t));
              q7_t *ptr;

              ptr=coefs.ptr();
              memcpy(coeffArray,ptr,this->nbTaps*sizeof(q7_t));
              this->pCoefs = coeffArray;
#else
              this->pCoefs=coefs.ptr();
#endif
              arm_fir_init_q7(&instFir,this->nbTaps,coefs.ptr(),state.ptr(),this->nbSamples);

              this->pSrc=samples.ptr();
              this->pDst=output.ptr();
           break;

           

          
       }
       
    }

    void FIRQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
