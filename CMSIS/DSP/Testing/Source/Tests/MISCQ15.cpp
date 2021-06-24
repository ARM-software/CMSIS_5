#include "arm_vec_math.h"

#include "MISCQ15.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 70
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)10)

#define ABS_ERROR_FAST_Q15 ((q15_t)20)


    void MISCQ15::test_correlate_q15()
    {
        const q15_t *inpA=inputA.ptr(); 
        const q15_t *inpB=inputB.ptr(); 
        q15_t *outp=output.ptr();

        arm_correlate_q15(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(q15_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR_Q15);

    }

    void MISCQ15::test_conv_q15()
    {
        const q15_t *inpA=inputA.ptr(); 
        const q15_t *inpB=inputB.ptr(); 
        q15_t *outp=output.ptr();

        arm_conv_q15(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(q15_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR_Q15);

    }

    // This value must be coherent with the Python script
    // generating the test patterns
    #define NBPOINTS 4

    void MISCQ15::test_conv_partial_q15()
    {
        const q15_t *inpA=inputA.ptr(); 
        const q15_t *inpB=inputB.ptr(); 
        q15_t *outp=output.ptr();
        q15_t *tmpp=tmp.ptr();


        arm_status status=arm_conv_partial_q15(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp,
          this->first,
          NBPOINTS);

 

        memcpy((void*)tmpp,(void*)&outp[this->first],NBPOINTS*sizeof(q15_t));
        ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        ASSERT_SNR(ref,tmp,(q15_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,tmp,ABS_ERROR_Q15);

    }

    void MISCQ15::test_conv_partial_fast_q15()
    {
        const q15_t *inpA=inputA.ptr(); 
        const q15_t *inpB=inputB.ptr(); 
        q15_t *outp=output.ptr();
        q15_t *tmpp=tmp.ptr();


        arm_status status=arm_conv_partial_fast_q15(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp,
          this->first,
          NBPOINTS);

 

        memcpy((void*)tmpp,(void*)&outp[this->first],NBPOINTS*sizeof(q15_t));
        ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        ASSERT_SNR(ref,tmp,(q15_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,tmp,ABS_ERROR_FAST_Q15);

    }

    void MISCQ15::test_conv_partial_opt_q15()
    {
        const q15_t *inpA=inputA.ptr(); 
        const q15_t *inpB=inputB.ptr(); 
        q15_t *outp=output.ptr();
        q15_t *tmpp=tmp.ptr();

        q15_t *scratchAp=scratchA.ptr();
        q15_t *scratchBp=scratchB.ptr();


        arm_status status=arm_conv_partial_opt_q15(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp,
          this->first,
          NBPOINTS,
          scratchAp,
          scratchBp
          );

 

        memcpy((void*)tmpp,(void*)&outp[this->first],NBPOINTS*sizeof(q15_t));
        ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        ASSERT_SNR(ref,tmp,(q15_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,tmp,ABS_ERROR_FAST_Q15);

    }

    void MISCQ15::test_conv_partial_fast_opt_q15()
    {
        const q15_t *inpA=inputA.ptr(); 
        const q15_t *inpB=inputB.ptr(); 
        q15_t *outp=output.ptr();
        q15_t *tmpp=tmp.ptr();

        q15_t *scratchAp=scratchA.ptr();
        q15_t *scratchBp=scratchB.ptr();



        arm_status status=arm_conv_partial_fast_opt_q15(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp,
          this->first,
          NBPOINTS,
          scratchAp,
          scratchBp);

 

        memcpy((void*)tmpp,(void*)&outp[this->first],NBPOINTS*sizeof(q15_t));
        ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        ASSERT_SNR(ref,tmp,(q15_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,tmp,ABS_ERROR_FAST_Q15);

    }


  
    void MISCQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {

            case MISCQ15::TEST_CORRELATE_Q15_1:
            {
                       this->nba = 14;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF1_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_2:
            {
                       this->nba = 14;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF2_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_3:
            {
                       this->nba = 14;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF3_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_4:
            {
                       this->nba = 14;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF4_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_5:
            {
                       this->nba = 14;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF5_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_6:
            {
                       this->nba = 15;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF6_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_7:
            {
                       this->nba = 15;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF7_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_8:
            {
                       this->nba = 15;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF8_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_9:
            {
                       this->nba = 15;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF9_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_10:
            {
                       this->nba = 15;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF10_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_11:
            {
                       this->nba = 16;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF11_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_12:
            {
                       this->nba = 16;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF12_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_13:
            {
                       this->nba = 16;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF13_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_14:
            {
                       this->nba = 16;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF14_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_15:
            {
                       this->nba = 16;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF15_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_16:
            {
                       this->nba = 17;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF16_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_17:
            {
                       this->nba = 17;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF17_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_18:
            {
                       this->nba = 17;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF18_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_19:
            {
                       this->nba = 17;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF19_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_20:
            {
                       this->nba = 17;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF20_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_21:
            {
                       this->nba = 32;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF21_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_22:
            {
                       this->nba = 32;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF22_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_23:
            {
                       this->nba = 32;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF23_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_24:
            {
                       this->nba = 32;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF24_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CORRELATE_Q15_25:
            {
                       this->nba = 32;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF25_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_26:
            {
                       this->nba = 14;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF26_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_27:
            {
                       this->nba = 14;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF27_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_28:
            {
                       this->nba = 14;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF28_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_29:
            {
                       this->nba = 14;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF29_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_30:
            {
                       this->nba = 14;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF30_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_31:
            {
                       this->nba = 15;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF31_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_32:
            {
                       this->nba = 15;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF32_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_33:
            {
                       this->nba = 15;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF33_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_34:
            {
                       this->nba = 15;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF34_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_35:
            {
                       this->nba = 15;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF35_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_36:
            {
                       this->nba = 16;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF36_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_37:
            {
                       this->nba = 16;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF37_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_38:
            {
                       this->nba = 16;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF38_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_39:
            {
                       this->nba = 16;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF39_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_40:
            {
                       this->nba = 16;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF40_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_41:
            {
                       this->nba = 17;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF41_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_42:
            {
                       this->nba = 17;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF42_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_43:
            {
                       this->nba = 17;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF43_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_44:
            {
                       this->nba = 17;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF44_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_45:
            {
                       this->nba = 17;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF45_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_46:
            {
                       this->nba = 32;
                       this->nbb = 15;
                       ref.reload(MISCQ15::REF46_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_47:
            {
                       this->nba = 32;
                       this->nbb = 16;
                       ref.reload(MISCQ15::REF47_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_48:
            {
                       this->nba = 32;
                       this->nbb = 17;
                       ref.reload(MISCQ15::REF48_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_49:
            {
                       this->nba = 32;
                       this->nbb = 18;
                       ref.reload(MISCQ15::REF49_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_Q15_50:
            {
                       this->nba = 32;
                       this->nbb = 33;
                       ref.reload(MISCQ15::REF50_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_PARTIAL_Q15_51:
            case MISCQ15::TEST_CONV_PARTIAL_FAST_Q15_54:
            case MISCQ15::TEST_CONV_PARTIAL_OPT_Q15_57:
            case MISCQ15::TEST_CONV_PARTIAL_FAST_OPT_Q15_60:
            {
              this->first=3;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCQ15::REF54_Q15_ID,mgr);
              tmp.create(ref.nbSamples(),MISCQ15::TMP_Q15_ID,mgr);

              // Oversized and not used except in opt case
              scratchA.create(24,MISCQ15::SCRATCH1_Q15_ID,mgr);
              scratchB.create(24,MISCQ15::SCRATCH2_Q15_ID,mgr);
            }
            break;

            case MISCQ15::TEST_CONV_PARTIAL_Q15_52:
            case MISCQ15::TEST_CONV_PARTIAL_FAST_Q15_55:
            case MISCQ15::TEST_CONV_PARTIAL_OPT_Q15_58:
            case MISCQ15::TEST_CONV_PARTIAL_FAST_OPT_Q15_61:
            {
              this->first=9;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCQ15::REF55_Q15_ID,mgr);
              tmp.create(ref.nbSamples(),MISCQ15::TMP_Q15_ID,mgr);

              // Oversized and not used except in opt case
              scratchA.create(24,MISCQ15::SCRATCH1_Q15_ID,mgr);
              scratchB.create(24,MISCQ15::SCRATCH2_Q15_ID,mgr);

            }
            break;

            case MISCQ15::TEST_CONV_PARTIAL_Q15_53:
            case MISCQ15::TEST_CONV_PARTIAL_FAST_Q15_56:
            case MISCQ15::TEST_CONV_PARTIAL_OPT_Q15_59:
            case MISCQ15::TEST_CONV_PARTIAL_FAST_OPT_Q15_62:
            {
              this->first=7;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCQ15::REF56_Q15_ID,mgr);
              tmp.create(ref.nbSamples(),MISCQ15::TMP_Q15_ID,mgr);

              // Oversized and not used except in opt case
              scratchA.create(24,MISCQ15::SCRATCH1_Q15_ID,mgr);
              scratchB.create(24,MISCQ15::SCRATCH2_Q15_ID,mgr);

            }
            break;


        }

       if (id >= MISCQ15::TEST_CONV_PARTIAL_Q15_51)
       {
          inputA.reload(MISCQ15::INPUTA2_Q15_ID,mgr,nba);
          inputB.reload(MISCQ15::INPUTB2_Q15_ID,mgr,nbb);
       }
       else
       {
          inputA.reload(MISCQ15::INPUTA_Q15_ID,mgr,nba);
          inputB.reload(MISCQ15::INPUTB_Q15_ID,mgr,nbb);
       }

       output.create(ref.nbSamples(),MISCQ15::OUT_Q15_ID,mgr);
        
    }

    void MISCQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
