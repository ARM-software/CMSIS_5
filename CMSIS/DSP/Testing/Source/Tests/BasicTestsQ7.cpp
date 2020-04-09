#include "BasicTestsQ7.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 20

#define ABS_ERROR_Q7 ((q7_t)2)
#define ABS_ERROR_Q31 ((q31_t)(1<<15))

#define ONEHALF 0x40

#define GET_Q7_PTR() \
const q7_t *inp1=input1.ptr(); \
const q7_t *inp2=input2.ptr(); \
q7_t *refp=ref.ptr(); \
q7_t *outp=output.ptr();

#define GET_LOGICAL_UINT8_PTR() \
const uint8_t *inp1=inputLogical1.ptr(); \
const uint8_t *inp2=inputLogical2.ptr(); \
uint8_t *refp=refLogical.ptr(); \
uint8_t *outp=outputLogical.ptr();

    void BasicTestsQ7::test_add_q7()
    {
        GET_Q7_PTR();

        arm_add_q7(inp1,inp2,outp,input1.nbSamples());
        
        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);

    } 

    void BasicTestsQ7::test_sub_q7()
    {
        GET_Q7_PTR();

        arm_sub_q7(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);
        
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
    } 

    void BasicTestsQ7::test_mult_q7()
    {
        GET_Q7_PTR();

        arm_mult_q7(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
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

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD - 1.0f);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
    } 

    void BasicTestsQ7::test_negate_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_negate_q7(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
    } 

    void BasicTestsQ7::test_offset_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_offset_q7(inp1,this->scalar,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
    } 

    void BasicTestsQ7::test_scale_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_scale_q7(inp1,this->scalar,0,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
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

        ASSERT_SNR(dotOutput,dotRef,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(dotOutput,dotRef,ABS_ERROR_Q31);

        ASSERT_EMPTY_TAIL(dotOutput);

       
    } 

    void BasicTestsQ7::test_abs_q7()
    {
        GET_Q7_PTR();

        arm_abs_q7(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
    } 

    void BasicTestsQ7::test_shift_q7()
    {
        const q7_t *inp1=input1.ptr();
        q7_t *refp=ref.ptr();
        q7_t *outp=output.ptr();

        arm_shift_q7(inp1,1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);
       
    } 

    void BasicTestsQ7::test_and_u8()
    {

            GET_LOGICAL_UINT8_PTR();


        arm_and_u8(inp1,inp2,outp,inputLogical1.nbSamples());
        
        ASSERT_EMPTY_TAIL(outputLogical);

        ASSERT_EQ(outputLogical, refLogical);


    } 

    void BasicTestsQ7::test_or_u8()
    {
        GET_LOGICAL_UINT8_PTR();

        arm_or_u8(inp1,inp2,outp,inputLogical1.nbSamples());
        
        ASSERT_EMPTY_TAIL(outputLogical);

        ASSERT_EQ(outputLogical, refLogical);

    } 

    void BasicTestsQ7::test_not_u8()
    {
        GET_LOGICAL_UINT8_PTR();

        arm_not_u8(inp1,outp,inputLogical1.nbSamples());
        
        ASSERT_EMPTY_TAIL(outputLogical);

        ASSERT_EQ(outputLogical, refLogical);

    } 

    void BasicTestsQ7::test_xor_u8()
    {
        GET_LOGICAL_UINT8_PTR();

        arm_xor_u8(inp1,inp2,outp,inputLogical1.nbSamples());
        
        ASSERT_EMPTY_TAIL(outputLogical);

        ASSERT_EQ(outputLogical, refLogical);

    } 

    void BasicTestsQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       this->scalar = ONEHALF;

       
       switch(id)
       {
        case BasicTestsQ7::TEST_ADD_Q7_1:
          nb = 15;
          ref.reload(BasicTestsQ7::REF_ADD_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_ADD_Q7_2:
          nb = 32;
          ref.reload(BasicTestsQ7::REF_ADD_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_ADD_Q7_3:
          nb = 47;
          ref.reload(BasicTestsQ7::REF_ADD_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;


        case BasicTestsQ7::TEST_SUB_Q7_4:
          nb = 15;
          ref.reload(BasicTestsQ7::REF_SUB_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SUB_Q7_5:
          nb = 32;
          ref.reload(BasicTestsQ7::REF_SUB_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SUB_Q7_6:
          nb = 47;
          ref.reload(BasicTestsQ7::REF_SUB_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_MULT_SHORT_Q7_7:
          nb = 15;
          ref.reload(BasicTestsQ7::REF_MULT_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_MULT_Q7_8:
          nb = 32;
          ref.reload(BasicTestsQ7::REF_MULT_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_MULT_Q7_9:
          nb = 47;
          ref.reload(BasicTestsQ7::REF_MULT_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_NEGATE_Q7_10:
          nb = 15;
          ref.reload(BasicTestsQ7::REF_NEGATE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_NEGATE_Q7_11:
          nb = 32;
          ref.reload(BasicTestsQ7::REF_NEGATE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_NEGATE_Q7_12:
          nb = 47;
          ref.reload(BasicTestsQ7::REF_NEGATE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_OFFSET_Q7_13:
          nb = 15;
          ref.reload(BasicTestsQ7::REF_OFFSET_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_OFFSET_Q7_14:
          nb = 32;
          ref.reload(BasicTestsQ7::REF_OFFSET_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_OFFSET_Q7_15:
          nb = 47;
          ref.reload(BasicTestsQ7::REF_OFFSET_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_SCALE_Q7_16:
          nb = 15;
          ref.reload(BasicTestsQ7::REF_SCALE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SCALE_Q7_17:
          nb = 32;
          ref.reload(BasicTestsQ7::REF_SCALE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_SCALE_Q7_18:
          nb = 47;
          ref.reload(BasicTestsQ7::REF_SCALE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_DOT_PROD_Q7_19:
          nb = 15;
          dotRef.reload(BasicTestsQ7::REF_DOT_3_Q7_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_DOT_PROD_Q7_20:
          nb = 32;
          dotRef.reload(BasicTestsQ7::REF_DOT_4N_Q7_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_DOT_PROD_Q7_21:
          nb = 47;
          dotRef.reload(BasicTestsQ7::REF_DOT_4N1_Q7_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_ABS_Q7_22:
          nb = 15;
          ref.reload(BasicTestsQ7::REF_ABS_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_ABS_Q7_23:
          nb = 32;
          ref.reload(BasicTestsQ7::REF_ABS_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_ABS_Q7_24:
          nb = 47;
          ref.reload(BasicTestsQ7::REF_ABS_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_ADD_Q7_25:
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          ref.reload(BasicTestsQ7::REF_POSSAT_12_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
        break;

        case BasicTestsQ7::TEST_ADD_Q7_26:
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          ref.reload(BasicTestsQ7::REF_NEGSAT_13_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SUB_Q7_27:
          ref.reload(BasicTestsQ7::REF_POSSAT_14_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SUB_Q7_28:
          ref.reload(BasicTestsQ7::REF_NEGSAT_15_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_MULT_Q7_29:
          ref.reload(BasicTestsQ7::REF_POSSAT_16_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
          input2.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_NEGATE_Q7_30:
          ref.reload(BasicTestsQ7::REF_POSSAT_17_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
          break;

        case BasicTestsQ7::TEST_OFFSET_Q7_31:
          ref.reload(BasicTestsQ7::REF_POSSAT_18_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
          /* 0.9 */
          this->scalar = 0x73;
          break;

        case BasicTestsQ7::TEST_OFFSET_Q7_32:
          ref.reload(BasicTestsQ7::REF_NEGSAT_19_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
          /* -0.9 */
          this->scalar = 0x8d;
          break;

        case BasicTestsQ7::TEST_SCALE_Q7_33:
          ref.reload(BasicTestsQ7::REF_POSSAT_20_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG2_Q7_ID,mgr);
          /* Minus max*/
          this->scalar = 0x80;
          break;

        case BasicTestsQ7::TEST_SHIFT_Q7_34:
          ref.reload(BasicTestsQ7::REF_SHIFT_21_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUTRAND_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SHIFT_Q7_35:
          ref.reload(BasicTestsQ7::REF_SHIFT_POSSAT_22_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXPOS_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_SHIFT_Q7_36:
          ref.reload(BasicTestsQ7::REF_SHIFT_NEGSAT_23_Q7_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::MAXNEG_Q7_ID,mgr);
        break;

        case BasicTestsQ7::TEST_AND_U8_37:
          nb = 15;
          refLogical.reload(BasicTestsQ7::REF_AND_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_AND_U8_38:
          nb = 32;
          refLogical.reload(BasicTestsQ7::REF_AND_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_AND_U8_39:
          nb = 47;
          refLogical.reload(BasicTestsQ7::REF_AND_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_OR_U8_40:
          nb = 15;
          refLogical.reload(BasicTestsQ7::REF_OR_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_OR_U8_41:
          nb = 32;
          refLogical.reload(BasicTestsQ7::REF_OR_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_OR_U8_42:
          nb = 47;
          refLogical.reload(BasicTestsQ7::REF_OR_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_NOT_U8_43:
          nb = 15;
          refLogical.reload(BasicTestsQ7::REF_NOT_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_NOT_U8_44:
          nb = 32;
          refLogical.reload(BasicTestsQ7::REF_NOT_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_NOT_U8_45:
          nb = 47;
          refLogical.reload(BasicTestsQ7::REF_NOT_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_XOR_U8_46:
          nb = 15;
          refLogical.reload(BasicTestsQ7::REF_XOR_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_XOR_U8_47:
          nb = 32;
          refLogical.reload(BasicTestsQ7::REF_XOR_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;
        case BasicTestsQ7::TEST_XOR_U8_48:
          nb = 47;
          refLogical.reload(BasicTestsQ7::REF_XOR_Q7_ID,mgr,nb);
          outputLogical.create(refLogical.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          inputLogical1.reload(BasicTestsQ7::INPUT1_BITWISE_Q7_ID,mgr,nb);
          inputLogical2.reload(BasicTestsQ7::INPUT2_BITWISE_Q7_ID,mgr,nb);
          break;

        case BasicTestsQ7::TEST_ADD_Q7_49:
          ref.reload(BasicTestsQ7::REF_ADD_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
        break;
        
        case BasicTestsQ7::TEST_SUB_Q7_50:
          ref.reload(BasicTestsQ7::REF_SUB_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
        break;
        
        case BasicTestsQ7::TEST_MULT_Q7_51:
          ref.reload(BasicTestsQ7::REF_MULT_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
        break;
        
        case BasicTestsQ7::TEST_NEGATE_Q7_52:
          ref.reload(BasicTestsQ7::REF_NEGATE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
        break;
        
        case BasicTestsQ7::TEST_OFFSET_Q7_53:
          ref.reload(BasicTestsQ7::REF_OFFSET_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
        break;
        
        case BasicTestsQ7::TEST_SCALE_Q7_54:
          ref.reload(BasicTestsQ7::REF_SCALE_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
        break;
        
        case BasicTestsQ7::TEST_DOT_PROD_Q7_55:
          dotRef.reload(BasicTestsQ7::REF_DOT_LONG_Q7_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
        break;
        
        case BasicTestsQ7::TEST_ABS_Q7_56:
          ref.reload(BasicTestsQ7::REF_ABS_Q7_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ7::OUT_SAMPLES_ID,mgr);
          input1.reload(BasicTestsQ7::INPUT1_Q7_ID,mgr,nb);
          input2.reload(BasicTestsQ7::INPUT2_Q7_ID,mgr,nb);
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
         case BasicTestsQ7::TEST_DOT_PROD_Q7_55:
            dotOutput.dump(mgr);
         break;

         case BasicTestsQ7::TEST_AND_U8_37:
         case BasicTestsQ7::TEST_AND_U8_38:
         case BasicTestsQ7::TEST_AND_U8_39:
         case BasicTestsQ7::TEST_OR_U8_40:
         case BasicTestsQ7::TEST_OR_U8_41:
         case BasicTestsQ7::TEST_OR_U8_42:
         case BasicTestsQ7::TEST_NOT_U8_43:
         case BasicTestsQ7::TEST_NOT_U8_44:
         case BasicTestsQ7::TEST_NOT_U8_45:
         case BasicTestsQ7::TEST_XOR_U8_46:
         case BasicTestsQ7::TEST_XOR_U8_47:
         case BasicTestsQ7::TEST_XOR_U8_48:
            outputLogical.dump(mgr);
         break;
         
         default:
            output.dump(mgr);
       }

        
    }
