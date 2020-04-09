#include "ControllerQ15.h"
#include "Error.h"

   
    void ControllerQ15::test_pid_q15()
    {
       for(int i=0; i < this->nbSamples; i++)
       {
          *this->pDst++ = arm_pid_q15(&instPid, *this->pSrc++);
       }
    } 
    
    void ControllerQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it;

       samples.reload(ControllerQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
       output.create(this->nbSamples,ControllerQ15::OUT_SAMPLES_Q15_ID,mgr);

       switch(id)
       {
           case TEST_PID_Q15_1:
              arm_pid_init_q15(&instPid,1);

              this->pSrc=samples.ptr();
              this->pDst=output.ptr();
           break;

       }
       
    }

    void ControllerQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
