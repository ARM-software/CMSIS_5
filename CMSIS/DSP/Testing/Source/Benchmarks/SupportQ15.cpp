#include "SupportQ15.h"
#include "Error.h"

   
    void SupportQ15::test_copy_q15()
    {
       
    } 

    void SupportQ15::test_fill_q15()
    {

    }

    void SupportQ15::test_q7_to_q15()
    {

    }

    void SupportQ15::test_q31_to_q15()
    {

    }


    void SupportQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       output.create(this->nbSamples,SupportQ15::OUT_SAMPLES_Q15_ID,mgr);

       switch(id)
       {
           case TEST_COPY_Q15_1:
           case TEST_FILL_Q15_2:
             samples.reload(SupportQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
             this->pSrc=samples.ptr();
           break;

           case TEST_Q31_TO_Q15_3:
             samplesQ31.reload(SupportQ15::SAMPLES_Q31_ID,mgr,this->nbSamples);
             this->pSrcQ31=samplesQ31.ptr();
           break;

           case TEST_Q7_TO_Q15_4:
             samplesQ7.reload(SupportQ15::SAMPLES_Q7_ID,mgr,this->nbSamples);
             this->pSrcQ7=samplesQ7.ptr();
           break;

       }
       this->pDst=output.ptr();
       
    }

    void SupportQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
