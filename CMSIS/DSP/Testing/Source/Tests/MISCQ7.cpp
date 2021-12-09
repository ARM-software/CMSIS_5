#include "arm_vec_math.h"

#include "MISCQ7.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 15
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q7 ((q7_t)5)
#define ABS_ERROR_FAST_Q7 ((q7_t)5)

    void MISCQ7::test_correlate_q7()
    {
        const q7_t *inpA=inputA.ptr(); 
        const q7_t *inpB=inputB.ptr(); 
        q7_t *outp=output.ptr();

        arm_correlate_q7(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(q7_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR_Q7);

    }

    void MISCQ7::test_conv_q7()
    {
        const q7_t *inpA=inputA.ptr(); 
        const q7_t *inpB=inputB.ptr(); 
        q7_t *outp=output.ptr();

        arm_conv_q7(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(q7_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR_Q7);

    }

    // This value must be coherent with the Python script
    // generating the test patterns
    #define NBPOINTS 4

    void MISCQ7::test_conv_partial_q7()
    {
        const q7_t *inpA=inputA.ptr(); 
        const q7_t *inpB=inputB.ptr(); 
        q7_t *outp=output.ptr();
        q7_t *tmpp=tmp.ptr();


        arm_status status=arm_conv_partial_q7(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp,
          this->first,
          NBPOINTS);

 

        memcpy((void*)tmpp,(void*)&outp[this->first],NBPOINTS*sizeof(q7_t));
        ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        ASSERT_SNR(ref,tmp,(q7_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,tmp,ABS_ERROR_Q7);

    }

    void MISCQ7::test_conv_partial_opt_q7()
    {
        const q7_t *inpA=inputA.ptr(); 
        const q7_t *inpB=inputB.ptr(); 
        q7_t *outp=output.ptr();
        q7_t *tmpp=tmp.ptr();

        q15_t *scratchAp=scratchA.ptr();
        q15_t *scratchBp=scratchB.ptr();


        arm_status status=arm_conv_partial_opt_q7(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp,
          this->first,
          NBPOINTS,
          scratchAp,
          scratchBp
          );

 

        memcpy((void*)tmpp,(void*)&outp[this->first],NBPOINTS*sizeof(q7_t));
        ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        ASSERT_SNR(ref,tmp,(q7_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,tmp,ABS_ERROR_FAST_Q7);

    }


  
    void MISCQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
           

            case MISCQ7::TEST_CORRELATE_Q7_1:
            {
                       this->nba = 30;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF1_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_2:
            {
                       this->nba = 30;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF2_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_3:
            {
                       this->nba = 30;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF3_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_4:
            {
                       this->nba = 30;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF4_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_5:
            {
                       this->nba = 30;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF5_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_6:
            {
                       this->nba = 31;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF6_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_7:
            {
                       this->nba = 31;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF7_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_8:
            {
                       this->nba = 31;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF8_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_9:
            {
                       this->nba = 31;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF9_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_10:
            {
                       this->nba = 31;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF10_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_11:
            {
                       this->nba = 32;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF11_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_12:
            {
                       this->nba = 32;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF12_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_13:
            {
                       this->nba = 32;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF13_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_14:
            {
                       this->nba = 32;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF14_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_15:
            {
                       this->nba = 32;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF15_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_16:
            {
                       this->nba = 33;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF16_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_17:
            {
                       this->nba = 33;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF17_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_18:
            {
                       this->nba = 33;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF18_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_19:
            {
                       this->nba = 33;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF19_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_20:
            {
                       this->nba = 33;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF20_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_21:
            {
                       this->nba = 48;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF21_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_22:
            {
                       this->nba = 48;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF22_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_23:
            {
                       this->nba = 48;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF23_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_24:
            {
                       this->nba = 48;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF24_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CORRELATE_Q7_25:
            {
                       this->nba = 48;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF25_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_26:
            {
                       this->nba = 30;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF26_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_27:
            {
                       this->nba = 30;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF27_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_28:
            {
                       this->nba = 30;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF28_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_29:
            {
                       this->nba = 30;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF29_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_30:
            {
                       this->nba = 30;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF30_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_31:
            {
                       this->nba = 31;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF31_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_32:
            {
                       this->nba = 31;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF32_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_33:
            {
                       this->nba = 31;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF33_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_34:
            {
                       this->nba = 31;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF34_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_35:
            {
                       this->nba = 31;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF35_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_36:
            {
                       this->nba = 32;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF36_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_37:
            {
                       this->nba = 32;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF37_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_38:
            {
                       this->nba = 32;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF38_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_39:
            {
                       this->nba = 32;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF39_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_40:
            {
                       this->nba = 32;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF40_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_41:
            {
                       this->nba = 33;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF41_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_42:
            {
                       this->nba = 33;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF42_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_43:
            {
                       this->nba = 33;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF43_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_44:
            {
                       this->nba = 33;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF44_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_45:
            {
                       this->nba = 33;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF45_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_46:
            {
                       this->nba = 48;
                       this->nbb = 31;
                       ref.reload(MISCQ7::REF46_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_47:
            {
                       this->nba = 48;
                       this->nbb = 32;
                       ref.reload(MISCQ7::REF47_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_48:
            {
                       this->nba = 48;
                       this->nbb = 33;
                       ref.reload(MISCQ7::REF48_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_49:
            {
                       this->nba = 48;
                       this->nbb = 34;
                       ref.reload(MISCQ7::REF49_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_Q7_50:
            {
                       this->nba = 48;
                       this->nbb = 49;
                       ref.reload(MISCQ7::REF50_Q7_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_PARTIAL_Q7_51:
            case MISCQ7::TEST_CONV_PARTIAL_OPT_Q7_54:
            {
              this->first=3;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCQ7::REF54_Q7_ID,mgr);
              tmp.create(ref.nbSamples(),MISCQ7::TMP_Q7_ID,mgr);

              // Oversized and not used except in opt case
              scratchA.create(24,MISCQ7::SCRATCH1_Q15_ID,mgr);
              scratchB.create(24,MISCQ7::SCRATCH2_Q15_ID,mgr);
            }
            break;

            case MISCQ7::TEST_CONV_PARTIAL_Q7_52:
            case MISCQ7::TEST_CONV_PARTIAL_OPT_Q7_55:
            {
              this->first=9;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCQ7::REF55_Q7_ID,mgr);
              tmp.create(ref.nbSamples(),MISCQ7::TMP_Q7_ID,mgr);

              // Oversized and not used except in opt case
              scratchA.create(24,MISCQ7::SCRATCH1_Q15_ID,mgr);
              scratchB.create(24,MISCQ7::SCRATCH2_Q15_ID,mgr);

            }
            break;

            case MISCQ7::TEST_CONV_PARTIAL_Q7_53:
            case MISCQ7::TEST_CONV_PARTIAL_OPT_Q7_56:
            {
              this->first=7;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCQ7::REF56_Q7_ID,mgr);
              tmp.create(ref.nbSamples(),MISCQ7::TMP_Q7_ID,mgr);

              // Oversized and not used except in opt case
              scratchA.create(24,MISCQ7::SCRATCH1_Q15_ID,mgr);
              scratchB.create(24,MISCQ7::SCRATCH2_Q15_ID,mgr);

            }
            break;


        }

       if (id >= MISCQ7::TEST_CONV_PARTIAL_Q7_51)
       {
          inputA.reload(MISCQ7::INPUTA2_Q7_ID,mgr,nba);
          inputB.reload(MISCQ7::INPUTB2_Q7_ID,mgr,nbb);
       }
       else
       {
          inputA.reload(MISCQ7::INPUTA_Q7_ID,mgr,nba);
          inputB.reload(MISCQ7::INPUTB_Q7_ID,mgr,nbb);
       }

       output.create(ref.nbSamples(),MISCQ7::OUT_Q7_ID,mgr);
        
    }

    void MISCQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
