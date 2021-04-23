#include "StatsF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void StatsF16::test_max_f16()
    {

        float16_t result;
        uint32_t  indexval;

        arm_max_f16(inap,
              this->nb,
              &result,
              &indexval);

    }

    void StatsF16::test_absmax_f16()
    {

        float16_t result;
        uint32_t  indexval;

        arm_absmax_f16(inap,
              this->nb,
              &result,
              &indexval);

    }

    void StatsF16::test_max_no_idx_f16()
    {
       
        float16_t result;

       
        arm_max_no_idx_f16(inap,
              this->nb,
              &result);

       

    }

    void StatsF16::test_min_f16()
    {
       
        float16_t result;
        uint32_t  indexval;

       
        arm_min_f16(inap,
              this->nb,
              &result,
              &indexval);

      

    }

    void StatsF16::test_absmin_f16()
    {
       
        float16_t result;
        uint32_t  indexval;

       
        arm_absmin_f16(inap,
              this->nb,
              &result,
              &indexval);

      

    }

    void StatsF16::test_mean_f16()
    {

        float16_t result;

        arm_mean_f16(inap,
              this->nb,
              &result);

    }

    void StatsF16::test_power_f16()
    {
        
        float16_t result;

        
        arm_power_f16(inap,
              this->nb,
              &result);

       

    }

    void StatsF16::test_rms_f16()
    {
       
        float16_t result;

       
        arm_rms_f16(inap,
              this->nb,
              &result);

       
    }

    void StatsF16::test_std_f16()
    {

        float16_t result;

       
        arm_std_f16(inap,
              this->nb,
              &result);

       
    }

    void StatsF16::test_var_f16()
    {

        float16_t result;

       
        arm_var_f16(inap,
              this->nb,
              &result);

      
    }


   

    void StatsF16::test_entropy_f16()
    {
      float16_t out;
      out = arm_entropy_f16(inap,this->nb);
      

    } 

    void StatsF16::test_logsumexp_f16()
    {
       float16_t out;

       out  = arm_logsumexp_f16(inap,this->nb);
     
    } 


    void StatsF16::test_kullback_leibler_f16()
    {
      
      float16_t out;

      out = arm_kullback_leibler_f16(inap,inbp,this->nb);
     
    } 

    void StatsF16::test_logsumexp_dot_prod_f16()
    {
      float16_t out;

      out = arm_logsumexp_dot_prod_f16(inap,inbp,this->nb,tmpp);
      
    } 

   
  
    void StatsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
        this->nb = *it;

        inputA.reload(StatsF16::INPUT1_F16_ID,mgr,this->nb);

        inap=inputA.ptr();

        switch(id)
        {
          case TEST_KULLBACK_LEIBLER_F16_10:
            inputB.reload(StatsF16::INPUT2_F16_ID,mgr,this->nb);

            inbp=inputB.ptr();
          break;

          case TEST_LOGSUMEXP_DOT_PROD_F16_11:
            inputB.reload(StatsF16::INPUT2_F16_ID,mgr,this->nb);

            inbp=inputB.ptr();

            tmp.create(this->nb,StatsF16::TMP_F16_ID,mgr);

            tmpp = tmp.ptr();

          break;

        }
        
    }

    void StatsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
    }
