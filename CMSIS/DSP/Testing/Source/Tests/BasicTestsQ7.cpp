#include "BasicTestsQ7.h"
#include "Error.h"

#define SNR_THRESHOLD 25

#define ONEHALF 0x40

#define GET_Q7_PTR() \
const q7_t *inp1=input1.ptr(); \
const q7_t *inp2=input2.ptr(); \
q7_t *refp=ref.ptr(); \
q7_t *outp=output.ptr();

    void BasicTestsQ7::test_add_q7()
    {
        GET_Q7_PTR();

        arm_add_q7(inp1,inp2,outp,input1.nbSamples());
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

    } 

    void BasicTestsQ7::test_sub_q7()
    {
        GET_Q7_PTR();

        arm_sub_q7(inp1,inp2,outp,input1.nbSamples());
        
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ7::test_mult_q7()
    {
        GET_Q7_PTR();

        arm_mult_q7(inp1,inp2,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    /*

    This test is run on a very short signal (3 samples).
    It is too short for a good SNR estimation.
    So, SNR is artificially decreased a little just for this test.

    */
    void BasicTestsQ7::test_mult_short_q7()
    {
        GET_Q7_PTR();

        arm_mult_q7(inp1,inp2,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD - 1.0);
       
    } 

    void BasicTestsQ7::test_negate_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_negate_q7(inp1,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ7::test_offset_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_offset_q7(inp1,this->scalar,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ7::test_scale_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_scale_q7(inp1,this->scalar,0,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ7::test_dot_prod_q7()
    {
        q31_t r;

        const q7_t *inp1=input1.ptr();
        const q7_t *inp2=input2.ptr();
        q31_t *refp=dotRef.ptr(); 
        q31_t *outp=dotOutput.ptr();

        arm_dot_prod_q7(inp1,inp2,input1.nbSamples(),&r);

        outp[0] = r;

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       
    } 

    void BasicTestsQ7::test_abs_q7()
    {
        GET_Q7_PTR();

        arm_abs_q7(inp1,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ7::test_shift_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_shift_q7(inp1,1,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 


    void BasicTestsQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       this->scalar = ONEHALF;

       
       switch(id)
       {
        case BasicTestsQ7::TEST_ADD_Q7_1:
          nb = 3;
          ref.reload(BasicTestsQ7::REF_ADD_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_ADD_Q7_2:
          nb = 8;
          ref.reload(BasicTestsQ7::REF_ADD_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_ADD_Q7_3:
          nb = 9;
          ref.reload(BasicTestsQ7::REF_ADD_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;


        case BasicTestsQ7::TEST_SUB_Q7_4:
          nb = 3;
          ref.reload(BasicTestsQ7::REF_SUB_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SUB_Q7_5:
          nb = 8;
          ref.reload(BasicTestsQ7::REF_SUB_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SUB_Q7_6:
          nb = 9;
          ref.reload(BasicTestsQ7::REF_SUB_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_MULT_SHORT_Q7_7:
          nb = 3;
          ref.reload(BasicTestsQ7::REF_MULT_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_MULT_Q7_8:
          nb = 8;
          ref.reload(BasicTestsQ7::REF_MULT_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_MULT_Q7_9:
          nb = 9;
          ref.reload(BasicTestsQ7::REF_MULT_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_NEGATE_Q7_10:
          nb = 3;
          ref.reload(BasicTestsQ7::REF_NEGATE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_NEGATE_Q7_11:
          nb = 8;
          ref.reload(BasicTestsQ7::REF_NEGATE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_NEGATE_Q7_12:
          nb = 9;
          ref.reload(BasicTestsQ7::REF_NEGATE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_OFFSET_Q7_13:
          nb = 3;
          ref.reload(BasicTestsQ7::REF_OFFSET_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_OFFSET_Q7_14:
          nb = 8;
          ref.reload(BasicTestsQ7::REF_OFFSET_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_OFFSET_Q7_15:
          nb = 9;
          ref.reload(BasicTestsQ7::REF_OFFSET_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_SCALE_Q7_16:
          nb = 3;
          ref.reload(BasicTestsQ7::REF_SCALE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SCALE_Q7_17:
          nb = 8;
          ref.reload(BasicTestsQ7::REF_SCALE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SCALE_Q7_18:
          nb = 9;
          ref.reload(BasicTestsQ7::REF_SCALE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_DOT_PROD_Q7_19:
          nb = 3;
          dotRef.reload(BasicTestsQ7::REF_DOT_3_Q7_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_DOT_PROD_Q7_20:
          nb = 8;
          dotRef.reload(BasicTestsQ7::REF_DOT_4N_Q7_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_DOT_PROD_Q7_21:
          nb = 9;
          dotRef.reload(BasicTestsQ7::REF_DOT_4N1_Q7_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_ABS_Q7_22:
          nb = 3;
          ref.reload(BasicTestsQ7::REF_ABS_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_ABS_Q7_23:
          nb = 8;
          ref.reload(BasicTestsQ7::REF_ABS_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_ABS_Q7_24:
          nb = 9;
          ref.reload(BasicTestsQ7::REF_ABS_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_ADD_Q7_25:
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          ref.reload(BasicTestsQ7::REF_POSSAT_12_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_ADD_Q7_26:
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          ref.reload(BasicTestsQ7::REF_NEGSAT_13_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SUB_Q7_27:
          ref.reload(BasicTestsQ7::REF_POSSAT_14_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SUB_Q7_28:
          ref.reload(BasicTestsQ7::REF_NEGSAT_15_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_MULT_Q7_29:
          ref.reload(BasicTestsQ7::REF_POSSAT_16_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_NEGATE_Q7_30:
          ref.reload(BasicTestsQ7::REF_POSSAT_17_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
          break;

        case BasicTestsQ7::TEST_OFFSET_Q7_31:
          ref.reload(BasicTestsQ7::REF_POSSAT_18_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          /* 0.9 */
          this->scalar = 0x73;
          break;

        case BasicTestsQ7::TEST_OFFSET_Q7_32:
          ref.reload(BasicTestsQ7::REF_NEGSAT_19_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          /* -0.9 */
          this->scalar = 0x8d;
          break;

        case BasicTestsQ7::TEST_SCALE_Q7_33:
          ref.reload(BasicTestsQ7::REF_POSSAT_20_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
          /* Minus max*/
          this->scalar = 0x80;
          break;

        case BasicTestsQ7::TEST_SHIFT_Q7_34:
          ref.reload(BasicTestsQ7::REF_SHIFT_21_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::INPUTRAND_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SHIFT_Q7_35:
          ref.reload(BasicTestsQ7::REF_SHIFT_POSSAT_22_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SHIFT_Q7_36:
          ref.reload(BasicTestsQ7::REF_SHIFT_NEGSAT_23_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
        break;

       }
      

       

    }

    void BasicTestsQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       switch(id)
       {
         case BasicTestsQ7::TEST_DOT_PROD_Q7_19:
         case BasicTestsQ7::TEST_DOT_PROD_Q7_20:
         case BasicTestsQ7::TEST_DOT_PROD_Q7_21:
            dotOutput.dump(mgr);
         break;

         default:
            output.dump(mgr);
       }

        
    }
