#include "SupportF32.h"
#include "Error.h"

   
    void SupportF32::test_copy_f32()
    {
       arm_copy_f32(this->pSrc,this->pDst,this->nbSamples);
    } 

    void SupportF32::test_fill_f32()
    {
       arm_fill_f32(0,this->pDst,this->nbSamples);
    }

    void SupportF32::test_q7_to_f32()
    {
      arm_q7_to_float(this->pSrcQ7,this->pDst,this->nbSamples);
    }

    void SupportF32::test_q15_to_f32()
    {
      arm_q15_to_float(this->pSrcQ15,this->pDst,this->nbSamples);
    }

    void SupportF32::test_q31_to_f32()
    {
      arm_q31_to_float(this->pSrcQ31,this->pDst,this->nbSamples);
    }

    void SupportF32::test_weighted_sum_f32()
    {
      arm_weighted_sum_f32(this->pSrc, this->pWeights,this->nbSamples);
    }

    void SupportF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       output.create(this->nbSamples,SupportF32::OUT_SAMPLES_F32_ID,mgr);

       switch(id)
       {
           case TEST_COPY_F32_1:
           case TEST_FILL_F32_2:
             samples.reload(SupportF32::SAMPLES_F32_ID,mgr,this->nbSamples);
             this->pSrc=samples.ptr();
           break;

           case TEST_Q15_TO_F32_3:
             samplesQ15.reload(SupportF32::SAMPLES_Q15_ID,mgr,this->nbSamples);
             this->pSrcQ15=samplesQ15.ptr();
           break;

           case TEST_Q31_TO_F32_4:
             samplesQ31.reload(SupportF32::SAMPLES_Q31_ID,mgr,this->nbSamples);
             this->pSrcQ31=samplesQ31.ptr();
           break;

           case TEST_Q7_TO_F32_5:
             samplesQ7.reload(SupportF32::SAMPLES_Q7_ID,mgr,this->nbSamples);
             this->pSrcQ7=samplesQ7.ptr();
           break;

           case TEST_WEIGHTED_SUM_F32_6:
              samples.reload(SupportF32::INPUTS6_F32_ID,mgr,this->nbSamples);
              weights.reload(SupportF32::WEIGHTS6_F32_ID,mgr,this->nbSamples);

              this->pSrc=samples.ptr();
              this->pWeights=weights.ptr();
           break;

       }

       this->pDst=output.ptr();
       
    }

    void SupportF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
