#include "FastMathQ31.h"
#include "Error.h"

   
    void FastMathQ31::test_cos_q31()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_cos_q31(*this->pSrc++);
       }
    } 

    void FastMathQ31::test_sin_q31()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_sin_q31(*this->pSrc++);
       }
    } 

    void FastMathQ31::test_sqrt_q31()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          arm_sqrt_q31(*this->pSrc++,this->pDst);
          this->pDst++;
       }
    } 

    void FastMathQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       samples.reload(FastMathQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
       output.create(this->nbSamples,FastMathQ31::OUT_SAMPLES_Q31_ID,mgr);


       this->pSrc=samples.ptr();
       this->pDst=output.ptr();
       
    }

    void FastMathQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
