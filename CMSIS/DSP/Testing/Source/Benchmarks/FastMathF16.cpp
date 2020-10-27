#include "FastMathF16.h"
#include "Error.h"

   #if 0
    void FastMathF16::test_cos_f16()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_cos_f16(*this->pSrc++);
       }
    } 

    void FastMathF16::test_sin_f16()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_sin_f16(*this->pSrc++);
       }
    } 
#endif

    void FastMathF16::test_sqrt_f16()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          arm_sqrt_f16(*this->pSrc++,this->pDst);
          this->pDst++;
       }
    } 

    void FastMathF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {

       (void)id;
       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       samples.reload(FastMathF16::SAMPLES_F16_ID,mgr,this->nbSamples);
       output.create(this->nbSamples,FastMathF16::OUT_SAMPLES_F16_ID,mgr);


       this->pSrc=samples.ptr();
       this->pDst=output.ptr();
       
    }

    void FastMathF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       (void)mgr;
    }
