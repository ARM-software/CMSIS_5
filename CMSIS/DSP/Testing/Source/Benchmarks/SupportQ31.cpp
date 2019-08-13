#include "SupportQ31.h"
#include "Error.h"

   
    void SupportQ31::test_copy_q31()
    {
       
    } 

    void SupportQ31::test_fill_q31()
    {

    }

    void SupportQ31::test_q7_to_q31()
    {

    }

    void SupportQ31::test_q15_to_q31()
    {

    }

    void SupportQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       output.create(this->nbSamples,SupportQ31::OUT_SAMPLES_Q31_ID,mgr);

       switch(id)
       {
           case TEST_COPY_Q31_1:
           case TEST_FILL_Q31_2:
             samples.reload(SupportQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
             this->pSrc=samples.ptr();
           break;

           case TEST_Q15_TO_Q31_3:
             samplesQ15.reload(SupportQ31::SAMPLES_Q15_ID,mgr,this->nbSamples);
             this->pSrcQ15=samplesQ15.ptr();
           break;

           case TEST_Q7_TO_Q31_4:
             samplesQ7.reload(SupportQ31::SAMPLES_Q7_ID,mgr,this->nbSamples);
             this->pSrcQ7=samplesQ7.ptr();
           break;

       }
       this->pDst=output.ptr();
       
    }

    void SupportQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
