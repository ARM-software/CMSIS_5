#include "StatsF32.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void StatsF32::test_max_f32()
    {

        float32_t result;
        uint32_t  indexval;

        arm_max_f32(inap,
              this->nb,
              &result,
              &indexval);

    }

    void StatsF32::test_absmax_f32()
    {

        float32_t result;
        uint32_t  indexval;

        arm_absmax_f32(inap,
              this->nb,
              &result,
              &indexval);

    }

    void StatsF32::test_max_no_idx_f32()
    {
       
        float32_t result;

       
        arm_max_no_idx_f32(inap,
              this->nb,
              &result);

       

    }

    void StatsF32::test_min_f32()
    {
       
        float32_t result;
        uint32_t  indexval;

       
        arm_min_f32(inap,
              this->nb,
              &result,
              &indexval);

      

    }

    void StatsF32::test_absmin_f32()
    {
       
        float32_t result;
        uint32_t  indexval;

       
        arm_absmin_f32(inap,
              this->nb,
              &result,
              &indexval);

      

    }

    void StatsF32::test_mean_f32()
    {

        float32_t result;

        arm_mean_f32(inap,
              this->nb,
              &result);

    }

    void StatsF32::test_power_f32()
    {
        
        float32_t result;

        
        arm_power_f32(inap,
              this->nb,
              &result);

       

    }

    void StatsF32::test_rms_f32()
    {
       
        float32_t result;

       
        arm_rms_f32(inap,
              this->nb,
              &result);

       
    }

    void StatsF32::test_std_f32()
    {

        float32_t result;

       
        arm_std_f32(inap,
              this->nb,
              &result);

       
    }

    void StatsF32::test_var_f32()
    {

        float32_t result;

       
        arm_var_f32(inap,
              this->nb,
              &result);

      
    }


   

    void StatsF32::test_entropy_f32()
    {
      float32_t out;
      out = arm_entropy_f32(inap,this->nb);
      

    } 

    void StatsF32::test_logsumexp_f32()
    {
       float32_t out;

       out  = arm_logsumexp_f32(inap,this->nb);
     
    } 


    void StatsF32::test_kullback_leibler_f32()
    {
      
      float32_t out;

      out = arm_kullback_leibler_f32(inap,inbp,this->nb);
     
    } 

    void StatsF32::test_logsumexp_dot_prod_f32()
    {
      float32_t out;

      out = arm_logsumexp_dot_prod_f32(inap,inbp,this->nb,tmpp);
      
    } 

   
  
    void StatsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
        this->nb = *it;

        inputA.reload(StatsF32::INPUT1_F32_ID,mgr,this->nb);

        inap=inputA.ptr();

        switch(id)
        {
          case TEST_KULLBACK_LEIBLER_F32_10:
            inputB.reload(StatsF32::INPUT2_F32_ID,mgr,this->nb);

            inbp=inputB.ptr();
          break;

          case TEST_LOGSUMEXP_DOT_PROD_F32_11:
            inputB.reload(StatsF32::INPUT2_F32_ID,mgr,this->nb);

            inbp=inputB.ptr();

            tmp.create(this->nb,StatsF32::TMP_F32_ID,mgr);

            tmpp = tmp.ptr();

          break;

        }
        
    }

    void StatsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
     
    }
