#include "SupportF16.h"
#include "Error.h"

   
    void SupportF16::test_copy_f16()
    {
       arm_copy_f16(this->pSrc,this->pDst,this->nbSamples);
    } 

    void SupportF16::test_fill_f16()
    {
       arm_fill_f16(0,this->pDst,this->nbSamples);
    }

    void SupportF16::test_q15_to_f16()
    {
      arm_q15_to_f16(this->pSrcQ15,this->pDst,this->nbSamples);
    }


    void SupportF16::test_f32_to_f16()
    {
      arm_float_to_f16(this->pSrcF32,this->pDst,this->nbSamples);
    }

    void SupportF16::test_weighted_sum_f16()
    {
      arm_weighted_sum_f16(this->pSrc, this->pWeights,this->nbSamples);
    }

    void SupportF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       output.create(this->nbSamples,SupportF16::OUT_SAMPLES_F16_ID,mgr);

       switch(id)
       {
           case TEST_COPY_F16_1:
           case TEST_FILL_F16_2:
             samples.reload(SupportF16::SAMPLES_F16_ID,mgr,this->nbSamples);
             this->pSrc=samples.ptr();
           break;

           case TEST_Q15_TO_F16_3:
             samplesQ15.reload(SupportF16::SAMPLES_Q15_ID,mgr,this->nbSamples);
             this->pSrcQ15=samplesQ15.ptr();
           break;

           case TEST_F32_TO_F16_4:
             samplesF32.reload(SupportF16::SAMPLES_F32_ID,mgr,this->nbSamples);
             this->pSrcF32=samplesF32.ptr();
           break;


           case TEST_WEIGHTED_SUM_F16_5:
              samples.reload(SupportF16::INPUTS6_F16_ID,mgr,this->nbSamples);
              weights.reload(SupportF16::WEIGHTS6_F16_ID,mgr,this->nbSamples);

              this->pSrc=samples.ptr();
              this->pWeights=weights.ptr();
           break;

       }

       this->pDst=output.ptr();
       
    }

    void SupportF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       (void)mgr;
    }
