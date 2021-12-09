#include "StatsQ7.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void StatsQ7::test_max_q7()
    {

        q7_t result;
        uint32_t  indexval;

        arm_max_q7(inap,
              this->nb,
              &result,
              &indexval);

    }

    void StatsQ7::test_absmax_q7()
    {

        q7_t result;
        uint32_t  indexval;

        arm_absmax_q7(inap,
              this->nb,
              &result,
              &indexval);

    }

   

    void StatsQ7::test_min_q7()
    {
       
        q7_t result;
        uint32_t  indexval;

       
        arm_min_q7(inap,
              this->nb,
              &result,
              &indexval);

      

    }

    void StatsQ7::test_absmin_q7()
    {
       
        q7_t result;
        uint32_t  indexval;

       
        arm_absmin_q7(inap,
              this->nb,
              &result,
              &indexval);

      

    }

    void StatsQ7::test_mean_q7()
    {
        q7_t result;

        arm_mean_q7(inap,
              this->nb,
              &result);

    }

    void StatsQ7::test_power_q7()
    {
        
        q31_t result;

        
        arm_power_q7(inap,
              this->nb,
              &result);

       

    }

   

   
  
    void StatsQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
        this->nb = *it;

        inputA.reload(StatsQ7::INPUT1_Q7_ID,mgr,this->nb);

        inap=inputA.ptr();

        
    }

    void StatsQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
     
    }
