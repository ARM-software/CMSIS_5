#include "StatsTestsF64.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 305
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (2.0e-15)

    void StatsTestsF64::test_max_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;
        uint32_t  indexval;

        float64_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        float64_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_max_f64(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF64::test_absmax_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;
        uint32_t  indexval;

        float64_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        float64_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_absmax_f64(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF64::test_max_no_idx_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_max_no_idx_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_EQ(result,refp[this->refOffset]);

    }

    void StatsTestsF64::test_absmax_no_idx_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_absmax_no_idx_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_EQ(result,refp[this->refOffset]);

    }

    void StatsTestsF64::test_min_no_idx_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_min_no_idx_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_EQ(result,refp[this->refOffset]);

    }

    void StatsTestsF64::test_absmin_no_idx_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_absmin_no_idx_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_EQ(result,refp[this->refOffset]);

    }

    void StatsTestsF64::test_min_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;
        uint32_t  indexval;

        float64_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        float64_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_min_f64(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF64::test_absmin_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;
        uint32_t  indexval;

        float64_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        float64_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_absmin_f64(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF64::test_mean_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_mean_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF64::test_power_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_power_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

/*
    void StatsTestsF64::test_rms_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_rms_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }
*/
    void StatsTestsF64::test_std_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_std_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF64::test_var_f64()
    {
        const float64_t *inp  = inputA.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_var_f64(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }


    void StatsTestsF64::test_std_stability_f64()
    {
      /*

      With the textbook algorithm, those values will produce a negative
      value for the variance.

      The CMSIS-DSP variance algorithm is the two pass one so will work
      with those values.

      So, it should be possible to compute the square root for the standard
      deviation.

      */
      float64_t in[4]={4.0f, 7.0f, 13.0f, 16.0f};
      float64_t result;
      int i;

      /*

      Add bigger offset so that average is much bigger than standard deviation.

      */
      for(i=0 ; i < 4; i++)
      {
        in[i] += 3.0e4f;
      }

      arm_std_f64(in,4,&result);

      /*

      If variance is giving a negative value, the square root
      should return zero.

      We check it is not happening here.


      */

      ASSERT_TRUE(fabs(5.47723f - result) < 1.0e-4f);

    }

    void StatsTestsF64::test_entropy_f64()
    {
      const float64_t *inp  = inputA.ptr();
      const int16_t *dimsp  = dims.ptr();

      float64_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_entropy_f64(inp,dimsp[i+1]);
         outp++;
         inp += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);

    } 
/*
    void StatsTestsF64::test_logsumexp_f64()
    {
      const float64_t *inp  = inputA.ptr();
      const int16_t *dimsp  = dims.ptr();

      float64_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_f64(inp,dimsp[i+1]);
         outp++;
         inp += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 

*/
    void StatsTestsF64::test_kullback_leibler_f64()
    {
      const float64_t *inpA  = inputA.ptr();
      const float64_t *inpB  = inputB.ptr();
      const int16_t *dimsp  = dims.ptr();

      float64_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_kullback_leibler_f64(inpA,inpB,dimsp[i+1]);
         outp++;
         inpA += dimsp[i+1];
         inpB += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 

/*
    void StatsTestsF64::test_logsumexp_dot_prod_f64()
    {
      const float64_t *inpA  = inputA.ptr();
      const float64_t *inpB  = inputB.ptr();
      const int16_t *dimsp  = dims.ptr();

      float64_t *outp         = output.ptr();
      float64_t *tmpp         = tmp.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_dot_prod_f64(inpA,inpB,dimsp[i+1],tmpp);
         outp++;
         inpA += dimsp[i+1];
         inpB += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 

*/
  
    void StatsTestsF64::test_mse_f64()
    {
        const float64_t *inpA  = inputA.ptr();
        const float64_t *inpB  = inputB.ptr();

        float64_t result;

        float64_t *refp  = ref.ptr();

        float64_t *outp  = output.ptr();

        arm_mse_f64(inpA,inpB,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],(float64_t)REL_ERROR);

    }

    void StatsTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case StatsTestsF64::TEST_MAX_F64_1:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               maxIndexes.reload(StatsTestsF64::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::MAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_MAX_F64_2:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               maxIndexes.reload(StatsTestsF64::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::MAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_MAX_F64_3:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               maxIndexes.reload(StatsTestsF64::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::MAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_MEAN_F64_4:
            {
               inputA.reload(StatsTestsF64::INPUT2_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::MEANVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_MEAN_F64_5:
            {
               inputA.reload(StatsTestsF64::INPUT2_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::MEANVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_MEAN_F64_6:
            {
               inputA.reload(StatsTestsF64::INPUT2_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::MEANVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_MIN_F64_7:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               minIndexes.reload(StatsTestsF64::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::MINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_MIN_F64_8:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               minIndexes.reload(StatsTestsF64::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::MINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_MIN_F64_9:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               minIndexes.reload(StatsTestsF64::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::MINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_POWER_F64_10:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::POWERVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_POWER_F64_11:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::POWERVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_POWER_F64_12:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::POWERVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_RMS_F64_13:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::RMSVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_RMS_F64_14:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::RMSVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_RMS_F64_15:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::RMSVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_STD_F64_16:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::STDVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_STD_F64_17:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::STDVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_STD_F64_18:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::STDVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_VAR_F64_19:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::VARVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_VAR_F64_20:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::VARVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_VAR_F64_21:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::VARVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_ENTROPY_F64_22:
            {
               inputA.reload(StatsTestsF64::INPUT22_F64_ID,mgr);
               dims.reload(StatsTestsF64::DIM22_S16_ID,mgr);
               ref.reload(StatsTestsF64::REF22_ENTROPY_F64_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF64::OUT_F64_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF64::TEST_LOGSUMEXP_F64_23:
            {
               inputA.reload(StatsTestsF64::INPUT23_F64_ID,mgr);
               dims.reload(StatsTestsF64::DIM23_S16_ID,mgr);
               ref.reload(StatsTestsF64::REF23_LOGSUMEXP_F64_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF64::OUT_F64_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF64::TEST_KULLBACK_LEIBLER_F64_24:
            {
               inputA.reload(StatsTestsF64::INPUTA24_F64_ID,mgr);
               inputB.reload(StatsTestsF64::INPUTB24_F64_ID,mgr);
               dims.reload(StatsTestsF64::DIM24_S16_ID,mgr);
               ref.reload(StatsTestsF64::REF24_KL_F64_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF64::OUT_F64_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF64::TEST_LOGSUMEXP_DOT_PROD_F64_25:
            {
               inputA.reload(StatsTestsF64::INPUTA25_F64_ID,mgr);
               inputB.reload(StatsTestsF64::INPUTB25_F64_ID,mgr);
               dims.reload(StatsTestsF64::DIM25_S16_ID,mgr);
               ref.reload(StatsTestsF64::REF25_LOGSUMEXP_DOT_F64_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF64::OUT_F64_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];

               /* 12 is max vecDim as defined in Python script generating the data */
               tmp.create(12,StatsTestsF64::TMP_F64_ID,mgr);
            }
            break;

            case StatsTestsF64::TEST_MAX_NO_IDX_F64_26:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::MAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_MAX_NO_IDX_F64_27:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::MAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_MAX_NO_IDX_F64_28:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::MAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case TEST_MEAN_F64_29:
               inputA.reload(StatsTestsF64::INPUT2_F64_ID,mgr,100);
              
               ref.reload(StatsTestsF64::MEANVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 3;
            break;

            case TEST_RMS_F64_30:
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,100);
              
               ref.reload(StatsTestsF64::RMSVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 3;
            break;

            case TEST_STD_F64_31:
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,100);
              
               ref.reload(StatsTestsF64::STDVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 3;
            break;

            case TEST_VAR_F64_32:
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,100);
              
               ref.reload(StatsTestsF64::VARVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 3;
            break;

            case StatsTestsF64::TEST_ABSMAX_F64_34:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,2);
              
               maxIndexes.reload(StatsTestsF64::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::ABSMAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_ABSMAX_F64_35:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,4);
              
               maxIndexes.reload(StatsTestsF64::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::ABSMAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_ABSMAX_F64_36:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,5);
              
               maxIndexes.reload(StatsTestsF64::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::ABSMAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_ABSMIN_F64_37:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,2);
              
               minIndexes.reload(StatsTestsF64::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::ABSMINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_ABSMIN_F64_38:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,4);
              
               minIndexes.reload(StatsTestsF64::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::ABSMINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_ABSMIN_F64_39:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,5);
              
               minIndexes.reload(StatsTestsF64::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF64::ABSMINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);
               index.create(1,StatsTestsF64::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_MIN_NO_IDX_F64_40:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::MINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_MIN_NO_IDX_F64_41:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::MINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_MIN_NO_IDX_F64_42:
            {
               inputA.reload(StatsTestsF64::INPUT1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::MINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;


            case StatsTestsF64::TEST_ABSMAX_NO_IDX_F64_43:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::ABSMAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_ABSMAX_NO_IDX_F64_44:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::ABSMAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_ABSMAX_NO_IDX_F64_45:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::ABSMAXVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_ABSMIN_NO_IDX_F64_46:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::ABSMINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_ABSMIN_NO_IDX_F64_47:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::ABSMINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_ABSMIN_NO_IDX_F64_48:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::ABSMINVALS_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_MSE_F64_49:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,2);
               inputB.reload(StatsTestsF64::INPUTNEW2_F64_ID,mgr,2);
              
               ref.reload(StatsTestsF64::MSE_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF64::TEST_MSE_F64_50:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,4);
               inputB.reload(StatsTestsF64::INPUTNEW2_F64_ID,mgr,4);
              
               ref.reload(StatsTestsF64::MSE_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF64::TEST_MSE_F64_51:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,5);
               inputB.reload(StatsTestsF64::INPUTNEW2_F64_ID,mgr,5);
              
               ref.reload(StatsTestsF64::MSE_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF64::TEST_MSE_F64_52:
            {
               inputA.reload(StatsTestsF64::INPUTNEW1_F64_ID,mgr,100);
               inputB.reload(StatsTestsF64::INPUTNEW2_F64_ID,mgr,100);
              
               ref.reload(StatsTestsF64::MSE_F64_ID,mgr);
               
               output.create(1,StatsTestsF64::OUT_F64_ID,mgr);

               refOffset = 3;
            }
            break;


        }
        
    }

    void StatsTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      switch(id)
      {
            case StatsTestsF64::TEST_MAX_F64_1:
            case StatsTestsF64::TEST_MAX_F64_2:
            case StatsTestsF64::TEST_MAX_F64_3:
            case StatsTestsF64::TEST_MIN_F64_7:
            case StatsTestsF64::TEST_MIN_F64_8:
            case StatsTestsF64::TEST_MIN_F64_9:
              index.dump(mgr);
              output.dump(mgr);
            break;

            default:
              output.dump(mgr);
      }
    }
