#include "StatsTestsF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 48
#define SNR_KULLBACK_THRESHOLD 40
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (6.0e-3)

#define REL_KULLBACK_ERROR (5.0e-3)
#define ABS_KULLBACK_ERROR (5.0e-3)

    void StatsTestsF16::test_max_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;
        uint32_t  indexval;

        float16_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        float16_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_max_f16(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF16::test_absmax_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;
        uint32_t  indexval;

        float16_t *refp  = ref.ptr();
        int16_t  *refind = maxIndexes.ptr();

        float16_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_absmax_f16(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }


    void StatsTestsF16::test_max_no_idx_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;

        float16_t *refp  = ref.ptr();

        float16_t *outp  = output.ptr();

        arm_max_no_idx_f16(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_EQ(result,refp[this->refOffset]);

    }


    void StatsTestsF16::test_min_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;
        uint32_t  indexval;

        float16_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        float16_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_min_f16(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF16::test_absmin_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;
        uint32_t  indexval;

        float16_t *refp  = ref.ptr();
        int16_t  *refind = minIndexes.ptr();

        float16_t *outp  = output.ptr();
        int16_t  *ind    = index.ptr();

        arm_absmin_f16(inp,
              inputA.nbSamples(),
              &result,
              &indexval);

        outp[0] = result;
        ind[0] = indexval;

        ASSERT_EQ(result,refp[this->refOffset]);
        ASSERT_EQ((int16_t)indexval,refind[this->refOffset]);

    }

    void StatsTestsF16::test_mean_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;

        float16_t *refp  = ref.ptr();

        float16_t *outp  = output.ptr();

        arm_mean_f16(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF16::test_power_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;

        float16_t *refp  = ref.ptr();

        float16_t *outp  = output.ptr();

        arm_power_f16(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF16::test_rms_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;

        float16_t *refp  = ref.ptr();

        float16_t *outp  = output.ptr();

        arm_rms_f16(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF16::test_std_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;

        float16_t *refp  = ref.ptr();

        float16_t *outp  = output.ptr();

        arm_std_f16(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }

    void StatsTestsF16::test_var_f16()
    {
        const float16_t *inp  = inputA.ptr();

        float16_t result;

        float16_t *refp  = ref.ptr();

        float16_t *outp  = output.ptr();

        arm_var_f16(inp,
              inputA.nbSamples(),
              &result);

        outp[0] = result;

        ASSERT_SNR(result,refp[this->refOffset],(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(result,refp[this->refOffset],REL_ERROR);

    }


    void StatsTestsF16::test_std_stability_f16()
    {
      /*

      With the textbook algorithm, those values will produce a negative
      value for the variance.

      The CMSIS-DSP variance algorithm is the two pass one so will work
      with those values.

      So, it should be possible to compute the square root for the standard
      deviation.

      */
      float16_t in[4]={4.0f, 7.0f, 13.0f, 16.0f};
      float16_t result;
      int i;

      /*

      Add bigger offset so that average is much bigger than standard deviation.

      */
      for(i=0 ; i < 4; i++)
      {
        in[i] += 3.0e3f;
      }

      arm_std_f16(in,4,&result);

      /*

      If variance is giving a negative value, the square root
      should return zero.

      We check it is not happening here.


      */
      ASSERT_TRUE(fabs(5.47723f - result) < 0.32f);

    }


    void StatsTestsF16::test_entropy_f16()
    {
      const float16_t *inp  = inputA.ptr();
      const int16_t *dimsp  = dims.ptr();

      float16_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_entropy_f16(inp,dimsp[i+1]);
         outp++;
         inp += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);

    } 

    void StatsTestsF16::test_logsumexp_f16()
    {
      const float16_t *inp  = inputA.ptr();
      const int16_t *dimsp  = dims.ptr();

      float16_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_f16(inp,dimsp[i+1]);
         outp++;
         inp += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 


    void StatsTestsF16::test_kullback_leibler_f16()
    {
      const float16_t *inpA  = inputA.ptr();
      const float16_t *inpB  = inputB.ptr();
      const int16_t *dimsp  = dims.ptr();

      float16_t *outp         = output.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_kullback_leibler_f16(inpA,inpB,dimsp[i+1]);
         outp++;
         inpA += dimsp[i+1];
         inpB += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float16_t)SNR_KULLBACK_THRESHOLD);

      ASSERT_CLOSE_ERROR(ref,output,ABS_KULLBACK_ERROR,REL_KULLBACK_ERROR);
    } 

    void StatsTestsF16::test_logsumexp_dot_prod_f16()
    {
      const float16_t *inpA  = inputA.ptr();
      const float16_t *inpB  = inputB.ptr();
      const int16_t *dimsp  = dims.ptr();

      float16_t *outp         = output.ptr();
      float16_t *tmpp         = tmp.ptr();

      for(int i=0;i < this->nbPatterns; i++)
      {
         *outp = arm_logsumexp_dot_prod_f16(inpA,inpB,dimsp[i+1],tmpp);
         outp++;
         inpA += dimsp[i+1];
         inpB += dimsp[i+1];
      }

      ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);

      ASSERT_REL_ERROR(ref,output,REL_ERROR);
    } 


  
    void StatsTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case StatsTestsF16::TEST_MAX_F16_1:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,7);
              
               maxIndexes.reload(StatsTestsF16::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::MAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_MAX_F16_2:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,16);
              
               maxIndexes.reload(StatsTestsF16::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::MAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_MAX_F16_3:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,23);
              
               maxIndexes.reload(StatsTestsF16::MAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::MAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_MEAN_F16_4:
            {
               inputA.reload(StatsTestsF16::INPUT2_F16_ID,mgr,7);
              
               ref.reload(StatsTestsF16::MEANVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_MEAN_F16_5:
            {
               inputA.reload(StatsTestsF16::INPUT2_F16_ID,mgr,16);
              
               ref.reload(StatsTestsF16::MEANVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_MEAN_F16_6:
            {
               inputA.reload(StatsTestsF16::INPUT2_F16_ID,mgr,23);
              
               ref.reload(StatsTestsF16::MEANVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_MIN_F16_7:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,7);
              
               minIndexes.reload(StatsTestsF16::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::MINVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_MIN_F16_8:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,16);
              
               minIndexes.reload(StatsTestsF16::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::MINVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_MIN_F16_9:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,23);
              
               minIndexes.reload(StatsTestsF16::MININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::MINVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_POWER_F16_10:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,7);
              
               ref.reload(StatsTestsF16::POWERVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_POWER_F16_11:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,16);
              
               ref.reload(StatsTestsF16::POWERVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_POWER_F16_12:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,23);
              
               ref.reload(StatsTestsF16::POWERVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_RMS_F16_13:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,7);
              
               ref.reload(StatsTestsF16::RMSVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_RMS_F16_14:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,16);
              
               ref.reload(StatsTestsF16::RMSVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_RMS_F16_15:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,23);
              
               ref.reload(StatsTestsF16::RMSVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_STD_F16_16:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,7);
              
               ref.reload(StatsTestsF16::STDVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_STD_F16_17:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,16);
              
               ref.reload(StatsTestsF16::STDVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_STD_F16_18:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,23);
              
               ref.reload(StatsTestsF16::STDVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_VAR_F16_19:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,7);
              
               ref.reload(StatsTestsF16::VARVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_VAR_F16_20:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,16);
              
               ref.reload(StatsTestsF16::VARVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_VAR_F16_21:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,23);
              
               ref.reload(StatsTestsF16::VARVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_ENTROPY_F16_22:
            {
               inputA.reload(StatsTestsF16::INPUT22_F16_ID,mgr);
               dims.reload(StatsTestsF16::DIM22_S16_ID,mgr);
               ref.reload(StatsTestsF16::REF22_ENTROPY_F16_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF16::OUT_F16_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF16::TEST_LOGSUMEXP_F16_23:
            {
               inputA.reload(StatsTestsF16::INPUT23_F16_ID,mgr);
               dims.reload(StatsTestsF16::DIM23_S16_ID,mgr);
               ref.reload(StatsTestsF16::REF23_LOGSUMEXP_F16_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF16::OUT_F16_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF16::TEST_KULLBACK_LEIBLER_F16_24:
            {
               inputA.reload(StatsTestsF16::INPUTA24_F16_ID,mgr);
               inputB.reload(StatsTestsF16::INPUTB24_F16_ID,mgr);
               dims.reload(StatsTestsF16::DIM24_S16_ID,mgr);
               ref.reload(StatsTestsF16::REF24_KL_F16_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF16::OUT_F16_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];
            }
            break;

            case StatsTestsF16::TEST_LOGSUMEXP_DOT_PROD_F16_25:
            {
               inputA.reload(StatsTestsF16::INPUTA25_F16_ID,mgr);
               inputB.reload(StatsTestsF16::INPUTB25_F16_ID,mgr);
               dims.reload(StatsTestsF16::DIM25_S16_ID,mgr);
               ref.reload(StatsTestsF16::REF25_LOGSUMEXP_DOT_F16_ID,mgr);
               output.create(ref.nbSamples(),StatsTestsF16::OUT_F16_ID,mgr);

               const int16_t *dimsp  = dims.ptr();
               this->nbPatterns=dimsp[0];

               /* 12 is max vecDim as defined in Python script generating the data */
               tmp.create(12,StatsTestsF16::TMP_F16_ID,mgr);
            }
            break;

            case StatsTestsF16::TEST_MAX_NO_IDX_F16_26:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,7);
              
               ref.reload(StatsTestsF16::MAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_MAX_NO_IDX_F16_27:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,8);
              
               ref.reload(StatsTestsF16::MAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_MAX_NO_IDX_F16_28:
            {
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,11);
              
               ref.reload(StatsTestsF16::MAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 2;
            }
            break;

            case TEST_MEAN_F16_29:
               inputA.reload(StatsTestsF16::INPUT2_F16_ID,mgr,100);
              
               ref.reload(StatsTestsF16::MEANVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 3;
            break;

            case TEST_RMS_F16_30:
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,100);
              
               ref.reload(StatsTestsF16::RMSVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 3;
            break;

            case TEST_STD_F16_31:
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,100);
              
               ref.reload(StatsTestsF16::STDVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 3;
            break;

            case TEST_VAR_F16_32:
               inputA.reload(StatsTestsF16::INPUT1_F16_ID,mgr,100);
              
               ref.reload(StatsTestsF16::VARVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);

               refOffset = 3;
            break;

            case StatsTestsF16::TEST_ABSMAX_F16_34:
            {
               inputA.reload(StatsTestsF16::INPUTNEW1_F16_ID,mgr,7);
              
               maxIndexes.reload(StatsTestsF16::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::ABSMAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_ABSMAX_F16_35:
            {
               inputA.reload(StatsTestsF16::INPUTNEW1_F16_ID,mgr,16);
              
               maxIndexes.reload(StatsTestsF16::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::ABSMAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_ABSMAX_F16_36:
            {
               inputA.reload(StatsTestsF16::INPUTNEW1_F16_ID,mgr,23);
              
               maxIndexes.reload(StatsTestsF16::ABSMAXINDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::ABSMAXVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;

            case StatsTestsF16::TEST_ABSMIN_F16_37:
            {
               inputA.reload(StatsTestsF16::INPUTNEW1_F16_ID,mgr,7);
              
               minIndexes.reload(StatsTestsF16::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::ABSMINVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 0;
            }
            break;

            case StatsTestsF16::TEST_ABSMIN_F16_38:
            {
               inputA.reload(StatsTestsF16::INPUTNEW1_F16_ID,mgr,16);
              
               minIndexes.reload(StatsTestsF16::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::ABSMINVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 1;
            }
            break;

            case StatsTestsF16::TEST_ABSMIN_F16_39:
            {
               inputA.reload(StatsTestsF16::INPUTNEW1_F16_ID,mgr,23);
              
               minIndexes.reload(StatsTestsF16::ABSMININDEXES_S16_ID,mgr);
               ref.reload(StatsTestsF16::ABSMINVALS_F16_ID,mgr);
               
               output.create(1,StatsTestsF16::OUT_F16_ID,mgr);
               index.create(1,StatsTestsF16::OUT_S16_ID,mgr);

               refOffset = 2;
            }
            break;
        }
        
    }

    void StatsTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      switch(id)
      {
            case StatsTestsF16::TEST_MAX_F16_1:
            case StatsTestsF16::TEST_MAX_F16_2:
            case StatsTestsF16::TEST_MAX_F16_3:
            case StatsTestsF16::TEST_MIN_F16_7:
            case StatsTestsF16::TEST_MIN_F16_8:
            case StatsTestsF16::TEST_MIN_F16_9:
              index.dump(mgr);
              output.dump(mgr);
            break;

            default:
              output.dump(mgr);
      }
    }
