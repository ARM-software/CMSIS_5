#include "SupportQ7.h"
#include "Error.h"

    void SupportQ7::test_copy_q7()
    {
       
    } 

    void SupportQ7::test_fill_q7()
    {

    }

    void SupportQ7::test_q15_to_q7()
    {

    }

    void SupportQ7::test_q31_to_q7()
    {

    }

    void SupportQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       output.create(this->nbSamples,SupportQ7::OUT_SAMPLES_Q7_ID,mgr);

       switch(id)
       {
           case TEST_COPY_Q7_1:
           case TEST_FILL_Q7_2:
             samples.reload(SupportQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
             this->pSrc=samples.ptr();
           break;

           case TEST_Q31_TO_Q7_3:
             samplesQ31.reload(SupportQ7::SAMPLES_Q31_ID,mgr,this->nbSamples);
             this->pSrcQ31=samplesQ31.ptr();
           break;

           case TEST_Q15_TO_Q7_4:
             samplesQ15.reload(SupportQ7::SAMPLES_Q15_ID,mgr,this->nbSamples);
             this->pSrcQ15=samplesQ15.ptr();
           break;

       }
       this->pDst=output.ptr();
       
    }

    void SupportQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
