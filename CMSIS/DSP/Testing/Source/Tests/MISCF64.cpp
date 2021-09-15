#include "arm_vec_math.h"

#include "MISCF64.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 310
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (2.0e-16)
#define ABS_ERROR (2.0e-16)

/*

For tests of the error value of the Levinson Durbin algorithm

*/
#define REL_LD_ERROR (1.0e-6)
#define ABS_LD_ERROR (1.0e-6)

/*
    void MISCF64::test_levinson_durbin_f64()
    {
        const float64_t *inpA=inputA.ptr(); 
        const float64_t *errs=inputB.ptr(); 
        float64_t *outp=output.ptr();
        float64_t err;

        float64_t refError=errs[this->errOffset];

        arm_levinson_durbin_f64(inpA,outp,&err,this->nba);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_LD_ERROR,REL_LD_ERROR);

        
        ASSERT_CLOSE_ERROR(refError,err,ABS_LD_ERROR,REL_LD_ERROR);

    }
*/
    void MISCF64::test_correlate_f64()
    {
        const float64_t *inpA=inputA.ptr(); 
        const float64_t *inpB=inputB.ptr(); 
        float64_t *outp=output.ptr();

        arm_correlate_f64(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }
/*
    void MISCF64::test_conv_f64()
    {
        const float64_t *inpA=inputA.ptr(); 
        const float64_t *inpB=inputB.ptr(); 
        float64_t *outp=output.ptr();

        arm_conv_f64(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

    // This value must be coherent with the Python script
    // generating the test patterns
    #define NBPOINTS 4

    void MISCF64::test_conv_partial_f64()
    {
        const float64_t *inpA=inputA.ptr(); 
        const float64_t *inpB=inputB.ptr(); 
        float64_t *outp=output.ptr();
        float64_t *tmpp=tmp.ptr();


        arm_status status=arm_conv_partial_f64(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp,
          this->first,
          NBPOINTS);

        memcpy((void*)tmpp,(void*)&outp[this->first],NBPOINTS*sizeof(float64_t));
        ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        ASSERT_SNR(ref,tmp,(float64_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,tmp,ABS_ERROR,REL_ERROR);


    }

*/
  
    void MISCF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {

            case MISCF64::TEST_CORRELATE_F64_1:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF1_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_2:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF2_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_3:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF3_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_4:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF4_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_5:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF5_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_6:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF6_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_7:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF7_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_8:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF8_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_9:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF9_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_10:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF10_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_11:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF11_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_12:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF12_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_13:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF13_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_14:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF14_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_15:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF15_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_16:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF16_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_17:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF17_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_18:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF18_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_19:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF19_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_20:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF20_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_21:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF21_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_22:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF22_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_23:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF23_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_24:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF24_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_25:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF25_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_26:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF26_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_27:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF27_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_28:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF28_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_29:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF29_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_30:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF30_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_31:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF31_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_32:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF32_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_33:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF33_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_34:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF34_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_35:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF35_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_36:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF36_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_37:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF37_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_38:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF38_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_39:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF39_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CORRELATE_F64_40:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF40_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_41:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF41_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_42:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF42_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_43:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF43_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_44:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF44_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_45:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF45_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_46:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF46_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_47:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF47_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_48:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF48_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_49:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF49_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_50:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF50_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_51:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF51_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_52:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF52_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_53:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF53_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_54:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF54_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_55:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF55_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_56:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF56_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_57:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF57_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_58:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF58_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_59:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF59_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_60:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF60_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_61:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF61_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_62:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF62_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_63:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF63_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_64:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF64_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_65:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF65_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_66:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF66_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_67:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF67_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_68:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF68_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_69:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF69_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_70:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF70_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_71:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF71_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_72:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF72_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_73:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF73_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_74:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF74_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_75:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF75_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_76:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCF64::REF76_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_77:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCF64::REF77_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_78:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCF64::REF78_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_79:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCF64::REF79_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_F64_80:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCF64::REF80_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_LEVINSON_DURBIN_F64_81:
            {
                       this->nba = 3;
                       inputA.reload(MISCF64::INPUTPHI_A_F64_ID,mgr);

                       this->errOffset=0;
                       inputB.reload(MISCF64::INPUT_ERRORS_F64_ID,mgr);
                       ref.reload(MISCF64::REF81_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_LEVINSON_DURBIN_F64_82:
            {
                       this->nba = 8;
                       inputA.reload(MISCF64::INPUTPHI_B_F64_ID,mgr);

                       this->errOffset=1;
                       inputB.reload(MISCF64::INPUT_ERRORS_F64_ID,mgr);
                       ref.reload(MISCF64::REF82_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_LEVINSON_DURBIN_F64_83:
            {
                       this->nba = 11;
                       inputA.reload(MISCF64::INPUTPHI_C_F64_ID,mgr);

                       this->errOffset=2;
                       inputB.reload(MISCF64::INPUT_ERRORS_F64_ID,mgr);
                       ref.reload(MISCF64::REF83_F64_ID,mgr);
            }
            break;

            case MISCF64::TEST_CONV_PARTIAL_F64_84:
            {
              this->first=3;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCF64::REF84_F64_ID,mgr);
              tmp.create(ref.nbSamples(),MISCF64::TMP_F64_ID,mgr);

            }
            break;

            case MISCF64::TEST_CONV_PARTIAL_F64_85:
            {
              this->first=9;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCF64::REF85_F64_ID,mgr);
              tmp.create(ref.nbSamples(),MISCF64::TMP_F64_ID,mgr);

            }
            break;

            case MISCF64::TEST_CONV_PARTIAL_F64_86:
            {
              this->first=7;
              this->nba = 6;
              this->nbb = 8;
              ref.reload(MISCF64::REF86_F64_ID,mgr);
              tmp.create(ref.nbSamples(),MISCF64::TMP_F64_ID,mgr);

            }
            break;


        }

       if (id < TEST_LEVINSON_DURBIN_F64_81) 
       {
         inputA.reload(MISCF64::INPUTA_F64_ID,mgr,nba);
         inputB.reload(MISCF64::INPUTB_F64_ID,mgr,nbb);
       }

       if (id > TEST_LEVINSON_DURBIN_F64_83)
       {
         inputA.reload(MISCF64::INPUTA2_F64_ID,mgr,nba);
         inputB.reload(MISCF64::INPUTB2_F64_ID,mgr,nbb);
       }

       output.create(ref.nbSamples(),MISCF64::OUT_F64_ID,mgr);
        
    }

    void MISCF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
