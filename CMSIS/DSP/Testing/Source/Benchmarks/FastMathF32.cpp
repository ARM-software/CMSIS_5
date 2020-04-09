#include "FastMathF32.h"
#include "Error.h"

   
    void FastMathF32::test_cos_f32()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_cos_f32(*this->pSrc++);
       }
    } 

    void FastMathF32::test_sin_f32()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_sin_f32(*this->pSrc++);
       }
    } 

    void FastMathF32::test_sqrt_f32()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          arm_sqrt_f32(*this->pSrc++,this->pDst);
          this->pDst++;
       }
    } 

    void FastMathF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       samples.reload(FastMathF32::SAMPLES_F32_ID,mgr,this->nbSamples);
       output.create(this->nbSamples,FastMathF32::OUT_SAMPLES_F32_ID,mgr);


       this->pSrc=samples.ptr();
       this->pDst=output.ptr();
       
    }

    void FastMathF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
