#include "MISCQ31.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "arm_vec_math.h"
#include "Test.h"

#define SNR_THRESHOLD 100
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)2)

    void MISCQ31::test_correlate_q31()
    {
        const q31_t *inpA=inputA.ptr(); 
        const q31_t *inpB=inputB.ptr(); 
        q31_t *outp=output.ptr();

        arm_correlate_q31(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(q31_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR_Q31);

    }

    void MISCQ31::test_conv_q31()
    {
        const q31_t *inpA=inputA.ptr(); 
        const q31_t *inpB=inputB.ptr(); 
        q31_t *outp=output.ptr();

        arm_conv_q31(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(q31_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(ref,output,ABS_ERROR_Q31);

    }


  
    void MISCQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        switch(id)
        {


            case MISCQ31::TEST_CORRELATE_Q31_1:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF1_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_2:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF2_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_3:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF3_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_4:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF4_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_5:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF5_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_6:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF6_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_7:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF7_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_8:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF8_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_9:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF9_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_10:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF10_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_11:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF11_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_12:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF12_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_13:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF13_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_14:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF14_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_15:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF15_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_16:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF16_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_17:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF17_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_18:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF18_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_19:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF19_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_20:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF20_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_21:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF21_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_22:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF22_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_23:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF23_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_24:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF24_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_25:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF25_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_26:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF26_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_27:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF27_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_28:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF28_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_29:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF29_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_30:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF30_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_31:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF31_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_32:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF32_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_33:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF33_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_34:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF34_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_35:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF35_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_36:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF36_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_37:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF37_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_38:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF38_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_39:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF39_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CORRELATE_Q31_40:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF40_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_41:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF41_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_42:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF42_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_43:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF43_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_44:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF44_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_45:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF45_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_46:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF46_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_47:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF47_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_48:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF48_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_49:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF49_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_50:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF50_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_51:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF51_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_52:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF52_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_53:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF53_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_54:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF54_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_55:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF55_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_56:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF56_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_57:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF57_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_58:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF58_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_59:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF59_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_60:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF60_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_61:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF61_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_62:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF62_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_63:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF63_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_64:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF64_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_65:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF65_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_66:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF66_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_67:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF67_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_68:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF68_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_69:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF69_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_70:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF70_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_71:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF71_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_72:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF72_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_73:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF73_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_74:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF74_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_75:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF75_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_76:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCQ31::REF76_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_77:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCQ31::REF77_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_78:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCQ31::REF78_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_79:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCQ31::REF79_Q31_ID,mgr);
            }
            break;

            case MISCQ31::TEST_CONV_Q31_80:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCQ31::REF80_Q31_ID,mgr);
            }
            break;

        }

       inputA.reload(MISCQ31::INPUTA_Q31_ID,mgr,nba);
       inputB.reload(MISCQ31::INPUTB_Q31_ID,mgr,nbb);

       output.create(ref.nbSamples(),MISCQ31::OUT_Q31_ID,mgr);
        
    }

    void MISCQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      output.dump(mgr);
      
    }
