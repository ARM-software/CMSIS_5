#include "MISCF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 60
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-4)
#define ABS_ERROR (1.0e-3)

/*

For tests of the error value of the Levinson Durbin algorithm

*/
#define SNR_LD_THRESHOLD 52
#define REL_LD_ERROR (1.0e-3)
#define ABS_LD_ERROR (1.0e-3)

    void MISCF16::test_levinson_durbin_f16()
    {
        const float16_t *inpA=inputA.ptr(); 
        const float16_t *errs=inputB.ptr(); 
        float16_t *outp=output.ptr();
        float16_t err;

        float16_t refError=errs[this->errOffset];

        arm_levinson_durbin_f16(inpA,outp,&err,this->nba);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(ref,output,(float16_t)SNR_LD_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_LD_ERROR,REL_LD_ERROR);

        
        ASSERT_CLOSE_ERROR(refError,err,ABS_LD_ERROR,REL_LD_ERROR);

    }

    void MISCF16::test_correlate_f16()
    {
        const float16_t *inpA=inputA.ptr(); 
        const float16_t *inpB=inputB.ptr(); 
        float16_t *outp=output.ptr();

        arm_correlate_f16(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

/*
    void MISCF16::test_conv_f16()
    {
        const float16_t *inpA=inputA.ptr(); 
        const float16_t *inpB=inputB.ptr(); 
        float16_t *outp=output.ptr();

        arm_conv_f16(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(float16_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }
*/

    

  
    void MISCF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {

            case MISCF16::TEST_CORRELATE_F16_1:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF1_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_2:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF2_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_3:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF3_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_4:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF4_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_5:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF5_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_6:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF6_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_7:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF7_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_8:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF8_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_9:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF9_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_10:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF10_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_11:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF11_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_12:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF12_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_13:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF13_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_14:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF14_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_15:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF15_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_16:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF16_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_17:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF17_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_18:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF18_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_19:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF19_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_20:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF20_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_21:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF21_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_22:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF22_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_23:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF23_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_24:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF24_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_25:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF25_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_26:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF26_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_27:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF27_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_28:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF28_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_29:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF29_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_30:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF30_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_31:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF31_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_32:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF32_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_33:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF33_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_34:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF34_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_35:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF35_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_36:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF36_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_37:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF37_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_38:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF38_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_39:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF39_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CORRELATE_F16_40:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF40_F16_ID,mgr);
            }
            break;

#if 0
            case MISCF16::TEST_CONV_F16_41:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF41_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_42:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF42_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_43:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF43_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_44:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF44_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_45:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF45_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_46:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF46_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_47:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF47_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_48:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF48_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_49:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF49_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_50:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF50_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_51:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF51_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_52:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF52_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_53:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF53_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_54:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF54_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_55:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF55_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_56:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF56_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_57:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF57_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_58:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF58_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_59:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF59_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_60:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF60_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_61:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF61_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_62:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF62_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_63:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF63_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_64:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF64_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_65:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF65_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_66:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF66_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_67:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF67_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_68:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF68_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_69:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF69_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_70:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF70_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_71:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF71_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_72:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF72_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_73:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF73_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_74:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF74_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_75:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF75_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_76:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCF16::REF76_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_77:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCF16::REF77_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_78:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCF16::REF78_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_79:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCF16::REF79_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_CONV_F16_80:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCF16::REF80_F16_ID,mgr);
            }
            break;
#endif

            case MISCF16::TEST_LEVINSON_DURBIN_F16_41:
            {
                       this->nba = 7;
                       inputA.reload(MISCF16::INPUTPHI_A_F16_ID,mgr);

                       this->errOffset=0;
                       inputB.reload(MISCF16::INPUT_ERRORS_F16_ID,mgr);
                       ref.reload(MISCF16::REF81_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_LEVINSON_DURBIN_F16_42:
            {
                       this->nba = 16;
                       inputA.reload(MISCF16::INPUTPHI_B_F16_ID,mgr);

                       this->errOffset=1;
                       inputB.reload(MISCF16::INPUT_ERRORS_F16_ID,mgr);
                       ref.reload(MISCF16::REF82_F16_ID,mgr);
            }
            break;

            case MISCF16::TEST_LEVINSON_DURBIN_F16_43:
            {
                       this->nba = 23;
                       inputA.reload(MISCF16::INPUTPHI_C_F16_ID,mgr);

                       this->errOffset=2;
                       inputB.reload(MISCF16::INPUT_ERRORS_F16_ID,mgr);
                       ref.reload(MISCF16::REF83_F16_ID,mgr);
            }
            break;

        }

       if (id < TEST_LEVINSON_DURBIN_F16_41)
       {
          inputA.reload(MISCF16::INPUTA_F16_ID,mgr,nba);
          inputB.reload(MISCF16::INPUTB_F16_ID,mgr,nbb);
       }

       output.create(ref.nbSamples(),MISCF16::OUT_F16_ID,mgr);
        
    }

    void MISCF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      output.dump(mgr);
      
    }
