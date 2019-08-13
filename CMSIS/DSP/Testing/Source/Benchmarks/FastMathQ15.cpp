#include "FastMathQ15.h"
#include "Error.h"

   
    void FastMathQ15::test_cos_q15()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_cos_q15(*this->pSrc++);
       }
    } 

    void FastMathQ15::test_sin_q15()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_sin_q15(*this->pSrc++);
       }
    } 

    void FastMathQ15::test_sqrt_q15()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          arm_sqrt_q15(*this->pSrc++,this->pDst);
          this->pDst++;
       }
    } 

    void FastMathQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       samples.reload(FastMathQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
       output.create(this->nbSamples,FastMathQ15::OUT_SAMPLES_Q15_ID,mgr);


       this->pSrc=samples.ptr();
       this->pDst=output.ptr();
       
    }

    void FastMathQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
