#include "StatsF64.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void StatsF64::test_entropy_f64()
    {
      float64_t out;
      out = arm_entropy_f64(inap,this->nb);
      

    } 

  

    void StatsF64::test_kullback_leibler_f64()
    {
      
      float64_t out;

      out = arm_kullback_leibler_f64(inap,inbp,this->nb);
     
    } 

   
  
    void StatsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
        this->nb = *it;

        inputA.reload(StatsF64::INPUT1_F64_ID,mgr,this->nb);

        inap=inputA.ptr();

        switch(id)
        {
          case TEST_KULLBACK_LEIBLER_F64_2:
            inputB.reload(StatsF64::INPUT2_F64_ID,mgr,this->nb);

            inbp=inputB.ptr();
          break;

          

        }
        
    }

    void StatsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
     
    }
