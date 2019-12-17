#include "MISCF32.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "arm_vec_math.h"
#include "Test.h"

#define SNR_THRESHOLD 120
/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)
#define ABS_ERROR (1.0e-5)

    void MISCF32::test_correlate_f32()
    {
        const float32_t *inpA=inputA.ptr(); 
        const float32_t *inpB=inputB.ptr(); 
        float32_t *outp=output.ptr();

        arm_correlate_f32(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }

    void MISCF32::test_conv_f32()
    {
        const float32_t *inpA=inputA.ptr(); 
        const float32_t *inpB=inputB.ptr(); 
        float32_t *outp=output.ptr();

        arm_conv_f32(inpA, inputA.nbSamples(),
          inpB, inputB.nbSamples(),
          outp);

        ASSERT_SNR(ref,output,(float32_t)SNR_THRESHOLD);
        ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);

    }


  
    void MISCF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        switch(id)
        {

            case MISCF32::TEST_CORRELATE_F32_1:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF1_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_2:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF2_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_3:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF3_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_4:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF4_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_5:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF5_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_6:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF6_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_7:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF7_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_8:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF8_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_9:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF9_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_10:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF10_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_11:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF11_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_12:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF12_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_13:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF13_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_14:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF14_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_15:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF15_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_16:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF16_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_17:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF17_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_18:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF18_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_19:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF19_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_20:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF20_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_21:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF21_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_22:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF22_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_23:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF23_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_24:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF24_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_25:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF25_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_26:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF26_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_27:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF27_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_28:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF28_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_29:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF29_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_30:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF30_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_31:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF31_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_32:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF32_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_33:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF33_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_34:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF34_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_35:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF35_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_36:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF36_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_37:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF37_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_38:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF38_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_39:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF39_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CORRELATE_F32_40:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF40_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_41:
            {
                       this->nba = 4;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF41_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_42:
            {
                       this->nba = 4;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF42_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_43:
            {
                       this->nba = 4;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF43_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_44:
            {
                       this->nba = 4;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF44_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_45:
            {
                       this->nba = 4;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF45_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_46:
            {
                       this->nba = 5;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF46_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_47:
            {
                       this->nba = 5;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF47_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_48:
            {
                       this->nba = 5;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF48_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_49:
            {
                       this->nba = 5;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF49_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_50:
            {
                       this->nba = 5;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF50_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_51:
            {
                       this->nba = 6;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF51_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_52:
            {
                       this->nba = 6;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF52_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_53:
            {
                       this->nba = 6;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF53_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_54:
            {
                       this->nba = 6;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF54_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_55:
            {
                       this->nba = 6;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF55_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_56:
            {
                       this->nba = 9;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF56_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_57:
            {
                       this->nba = 9;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF57_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_58:
            {
                       this->nba = 9;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF58_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_59:
            {
                       this->nba = 9;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF59_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_60:
            {
                       this->nba = 9;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF60_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_61:
            {
                       this->nba = 10;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF61_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_62:
            {
                       this->nba = 10;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF62_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_63:
            {
                       this->nba = 10;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF63_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_64:
            {
                       this->nba = 10;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF64_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_65:
            {
                       this->nba = 10;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF65_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_66:
            {
                       this->nba = 11;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF66_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_67:
            {
                       this->nba = 11;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF67_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_68:
            {
                       this->nba = 11;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF68_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_69:
            {
                       this->nba = 11;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF69_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_70:
            {
                       this->nba = 11;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF70_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_71:
            {
                       this->nba = 12;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF71_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_72:
            {
                       this->nba = 12;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF72_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_73:
            {
                       this->nba = 12;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF73_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_74:
            {
                       this->nba = 12;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF74_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_75:
            {
                       this->nba = 12;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF75_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_76:
            {
                       this->nba = 13;
                       this->nbb = 1;
                       ref.reload(MISCF32::REF76_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_77:
            {
                       this->nba = 13;
                       this->nbb = 2;
                       ref.reload(MISCF32::REF77_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_78:
            {
                       this->nba = 13;
                       this->nbb = 3;
                       ref.reload(MISCF32::REF78_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_79:
            {
                       this->nba = 13;
                       this->nbb = 8;
                       ref.reload(MISCF32::REF79_F32_ID,mgr);
            }
            break;

            case MISCF32::TEST_CONV_F32_80:
            {
                       this->nba = 13;
                       this->nbb = 11;
                       ref.reload(MISCF32::REF80_F32_ID,mgr);
            }
            break;


        }

       inputA.reload(MISCF32::INPUTA_F32_ID,mgr,nba);
       inputB.reload(MISCF32::INPUTB_F32_ID,mgr,nbb);

       output.create(ref.nbSamples(),MISCF32::OUT_F32_ID,mgr);
        
    }

    void MISCF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      output.dump(mgr);
      
    }
