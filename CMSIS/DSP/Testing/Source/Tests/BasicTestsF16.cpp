#include "BasicTestsF16.h"
#include <stdio.h>
#include "Error.h"

#include "arm_math_f16.h"

#define SNR_THRESHOLD 62
#define SNR_DOTPROD_THRESHOLD 40

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (4e-2)


#define GET_F16_PTR() \
const float16_t *inp1=input1.ptr(); \
const float16_t *inp2=input2.ptr(); \
float16_t *refp=ref.ptr(); \
float16_t *outp=output.ptr();

    void BasicTestsF16::test_add_f16()
    {
        GET_F16_PTR();

        arm_add_f16(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

    void BasicTestsF16::test_sub_f16()
    {

        GET_F16_PTR();

        arm_sub_f16(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);
        
        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
 
    } 

    void BasicTestsF16::test_mult_f16()
    {

        GET_F16_PTR();

        arm_mult_f16(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
   
    } 

    void BasicTestsF16::test_negate_f16()
    {

        GET_F16_PTR();

        arm_negate_f16(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
  
    } 

    void BasicTestsF16::test_offset_f16()
    {

        GET_F16_PTR();

        arm_offset_f16(inp1,0.5,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
  
    } 

    void BasicTestsF16::test_scale_f16()
    {

        GET_F16_PTR();

        arm_scale_f16(inp1,0.5,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
   
    } 

    void BasicTestsF16::test_dot_prod_f16()
    {

        float16_t r;

        GET_F16_PTR();

        arm_dot_prod_f16(inp1,inp2,input1.nbSamples(),&r);

        outp[0] = r;

        ASSERT_SNR(output,ref,(float16_t)SNR_DOTPROD_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

        ASSERT_EMPTY_TAIL(output);

       
    } 

    void BasicTestsF16::test_abs_f16()
    {

        GET_F16_PTR();

        arm_abs_f16(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
 
    } 

 
    void BasicTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case BasicTestsF16::TEST_ADD_F16_1:
          nb = 7;
          ref.reload(BasicTestsF16::REF_ADD_F16_ID,mgr,nb);
          break;

        case BasicTestsF16::TEST_ADD_F16_2:
          nb = 16;
          ref.reload(BasicTestsF16::REF_ADD_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_ADD_F16_3:
          nb = 23;
          ref.reload(BasicTestsF16::REF_ADD_F16_ID,mgr,nb);
          break;


        case BasicTestsF16::TEST_SUB_F16_4:
          nb = 7;
          ref.reload(BasicTestsF16::REF_SUB_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_SUB_F16_5:
          nb = 16;
          ref.reload(BasicTestsF16::REF_SUB_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_SUB_F16_6:
          nb = 23;
          ref.reload(BasicTestsF16::REF_SUB_F16_ID,mgr,nb);
          break;

        case BasicTestsF16::TEST_MULT_F16_7:
          nb = 7;
          ref.reload(BasicTestsF16::REF_MULT_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_MULT_F16_8:
          nb = 16;
          ref.reload(BasicTestsF16::REF_MULT_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_MULT_F16_9:
          nb = 23;
          ref.reload(BasicTestsF16::REF_MULT_F16_ID,mgr,nb);
          break;

        case BasicTestsF16::TEST_NEGATE_F16_10:
          nb = 7;
          ref.reload(BasicTestsF16::REF_NEGATE_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_NEGATE_F16_11:
          nb = 16;
          ref.reload(BasicTestsF16::REF_NEGATE_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_NEGATE_F16_12:
          nb = 23;
          ref.reload(BasicTestsF16::REF_NEGATE_F16_ID,mgr,nb);
          break;

        case BasicTestsF16::TEST_OFFSET_F16_13:
          nb = 7;
          ref.reload(BasicTestsF16::REF_OFFSET_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_OFFSET_F16_14:
          nb = 16;
          ref.reload(BasicTestsF16::REF_OFFSET_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_OFFSET_F16_15:
          nb = 23;
          ref.reload(BasicTestsF16::REF_OFFSET_F16_ID,mgr,nb);
          break;

        case BasicTestsF16::TEST_SCALE_F16_16:
          nb = 7;
          ref.reload(BasicTestsF16::REF_SCALE_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_SCALE_F16_17:
          nb = 16;
          ref.reload(BasicTestsF16::REF_SCALE_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_SCALE_F16_18:
          nb = 23;
          ref.reload(BasicTestsF16::REF_SCALE_F16_ID,mgr,nb);
          break;

        case BasicTestsF16::TEST_DOT_PROD_F16_19:
          nb = 7;
          ref.reload(BasicTestsF16::REF_DOT_3_F16_ID,mgr);
          break;
        case BasicTestsF16::TEST_DOT_PROD_F16_20:
          nb = 16;
          ref.reload(BasicTestsF16::REF_DOT_4N_F16_ID,mgr);
          break;
        case BasicTestsF16::TEST_DOT_PROD_F16_21:
          nb = 23;
          ref.reload(BasicTestsF16::REF_DOT_4N1_F16_ID,mgr);
          break;

        case BasicTestsF16::TEST_ABS_F16_22:
          nb = 7;
          ref.reload(BasicTestsF16::REF_ABS_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_ABS_F16_23:
          nb = 16;
          ref.reload(BasicTestsF16::REF_ABS_F16_ID,mgr,nb);
          break;
        case BasicTestsF16::TEST_ABS_F16_24:
          nb = 23;
          ref.reload(BasicTestsF16::REF_ABS_F16_ID,mgr,nb);
          break;

        case BasicTestsF16::TEST_ADD_F16_25:
          ref.reload(BasicTestsF16::REF_ADD_F16_ID,mgr,nb);
        break;

        case BasicTestsF16::TEST_SUB_F16_26:
          ref.reload(BasicTestsF16::REF_SUB_F16_ID,mgr,nb);
        break;

        case BasicTestsF16::TEST_MULT_F16_27:
          ref.reload(BasicTestsF16::REF_MULT_F16_ID,mgr,nb);
        break;

        case BasicTestsF16::TEST_NEGATE_F16_28:
          ref.reload(BasicTestsF16::REF_NEGATE_F16_ID,mgr,nb);
        break;

        case BasicTestsF16::TEST_OFFSET_F16_29:
          ref.reload(BasicTestsF16::REF_OFFSET_F16_ID,mgr,nb);
        break;

        case BasicTestsF16::TEST_SCALE_F16_30:
          ref.reload(BasicTestsF16::REF_SCALE_F16_ID,mgr,nb);
        break;

        case BasicTestsF16::TEST_DOT_PROD_F16_31:
          ref.reload(BasicTestsF16::REF_DOT_LONG_F16_ID,mgr);
        break;

        case BasicTestsF16::TEST_ABS_F16_32:
          ref.reload(BasicTestsF16::REF_ABS_F16_ID,mgr,nb);
        break;

       }
      

       input1.reload(BasicTestsF16::INPUT1_F16_ID,mgr,nb);
       input2.reload(BasicTestsF16::INPUT2_F16_ID,mgr,nb);

       output.create(ref.nbSamples(),BasicTestsF16::OUT_SAMPLES_F16_ID,mgr);
    }

    void BasicTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
