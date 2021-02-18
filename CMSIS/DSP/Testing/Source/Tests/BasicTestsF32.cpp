#include "BasicTestsF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (5.0e-5)

#define GET_F32_PTR() \
const float32_t *inp1=input1.ptr(); \
const float32_t *inp2=input2.ptr(); \
float32_t *outp=output.ptr();

    void BasicTestsF32::test_add_f32()
    {
        GET_F32_PTR();

        arm_add_f32(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

    void BasicTestsF32::test_clip_f32()
    {
        const float32_t *inp=input1.ptr();
        float32_t *outp=output.ptr();

        arm_clip_f32(inp,outp,this->min, this->max,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

    void BasicTestsF32::test_sub_f32()
    {
        GET_F32_PTR();

        arm_sub_f32(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);
        
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF32::test_mult_f32()
    {
        GET_F32_PTR();

        arm_mult_f32(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF32::test_negate_f32()
    {
        GET_F32_PTR();

        (void)inp2;

        arm_negate_f32(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF32::test_offset_f32()
    {
        GET_F32_PTR();

        (void)inp2;

        arm_offset_f32(inp1,0.5,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF32::test_scale_f32()
    {
        GET_F32_PTR();

        (void)inp2;

        arm_scale_f32(inp1,0.5,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF32::test_dot_prod_f32()
    {
        float32_t r;

        GET_F32_PTR();

        arm_dot_prod_f32(inp1,inp2,input1.nbSamples(),&r);

        outp[0] = r;

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

        ASSERT_EMPTY_TAIL(output);

       
    } 

    void BasicTestsF32::test_abs_f32()
    {
        GET_F32_PTR();

        (void)inp2;

        arm_abs_f32(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

 
    void BasicTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;

       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case BasicTestsF32::TEST_ADD_F32_1:
          nb = 3;
          ref.reload(BasicTestsF32::REF_ADD_F32_ID,mgr,nb);
          break;

        case BasicTestsF32::TEST_ADD_F32_2:
          nb = 8;
          ref.reload(BasicTestsF32::REF_ADD_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_ADD_F32_3:
          nb = 11;
          ref.reload(BasicTestsF32::REF_ADD_F32_ID,mgr,nb);
          break;


        case BasicTestsF32::TEST_SUB_F32_4:
          nb = 3;
          ref.reload(BasicTestsF32::REF_SUB_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_SUB_F32_5:
          nb = 8;
          ref.reload(BasicTestsF32::REF_SUB_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_SUB_F32_6:
          nb = 11;
          ref.reload(BasicTestsF32::REF_SUB_F32_ID,mgr,nb);
          break;

        case BasicTestsF32::TEST_MULT_F32_7:
          nb = 3;
          ref.reload(BasicTestsF32::REF_MULT_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_MULT_F32_8:
          nb = 8;
          ref.reload(BasicTestsF32::REF_MULT_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_MULT_F32_9:
          nb = 11;
          ref.reload(BasicTestsF32::REF_MULT_F32_ID,mgr,nb);
          break;

        case BasicTestsF32::TEST_NEGATE_F32_10:
          nb = 3;
          ref.reload(BasicTestsF32::REF_NEGATE_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_NEGATE_F32_11:
          nb = 8;
          ref.reload(BasicTestsF32::REF_NEGATE_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_NEGATE_F32_12:
          nb = 11;
          ref.reload(BasicTestsF32::REF_NEGATE_F32_ID,mgr,nb);
          break;

        case BasicTestsF32::TEST_OFFSET_F32_13:
          nb = 3;
          ref.reload(BasicTestsF32::REF_OFFSET_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_OFFSET_F32_14:
          nb = 8;
          ref.reload(BasicTestsF32::REF_OFFSET_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_OFFSET_F32_15:
          nb = 11;
          ref.reload(BasicTestsF32::REF_OFFSET_F32_ID,mgr,nb);
          break;

        case BasicTestsF32::TEST_SCALE_F32_16:
          nb = 3;
          ref.reload(BasicTestsF32::REF_SCALE_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_SCALE_F32_17:
          nb = 8;
          ref.reload(BasicTestsF32::REF_SCALE_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_SCALE_F32_18:
          nb = 11;
          ref.reload(BasicTestsF32::REF_SCALE_F32_ID,mgr,nb);
          break;

        case BasicTestsF32::TEST_DOT_PROD_F32_19:
          nb = 3;
          ref.reload(BasicTestsF32::REF_DOT_3_F32_ID,mgr);
          break;
        case BasicTestsF32::TEST_DOT_PROD_F32_20:
          nb = 8;
          ref.reload(BasicTestsF32::REF_DOT_4N_F32_ID,mgr);
          break;
        case BasicTestsF32::TEST_DOT_PROD_F32_21:
          nb = 11;
          ref.reload(BasicTestsF32::REF_DOT_4N1_F32_ID,mgr);
          break;

        case BasicTestsF32::TEST_ABS_F32_22:
          nb = 3;
          ref.reload(BasicTestsF32::REF_ABS_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_ABS_F32_23:
          nb = 8;
          ref.reload(BasicTestsF32::REF_ABS_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_ABS_F32_24:
          nb = 11;
          ref.reload(BasicTestsF32::REF_ABS_F32_ID,mgr,nb);
          break;

        case BasicTestsF32::TEST_ADD_F32_25:
          ref.reload(BasicTestsF32::REF_ADD_F32_ID,mgr,nb);
        break;

        case BasicTestsF32::TEST_SUB_F32_26:
          ref.reload(BasicTestsF32::REF_SUB_F32_ID,mgr,nb);
        break;

        case BasicTestsF32::TEST_MULT_F32_27:
          ref.reload(BasicTestsF32::REF_MULT_F32_ID,mgr,nb);
        break;

        case BasicTestsF32::TEST_NEGATE_F32_28:
          ref.reload(BasicTestsF32::REF_NEGATE_F32_ID,mgr,nb);
        break;

        case BasicTestsF32::TEST_OFFSET_F32_29:
          ref.reload(BasicTestsF32::REF_OFFSET_F32_ID,mgr,nb);
        break;

        case BasicTestsF32::TEST_SCALE_F32_30:
          ref.reload(BasicTestsF32::REF_SCALE_F32_ID,mgr,nb);
        break;

        case BasicTestsF32::TEST_DOT_PROD_F32_31:
          ref.reload(BasicTestsF32::REF_DOT_LONG_F32_ID,mgr);
        break;

        case BasicTestsF32::TEST_ABS_F32_32:
          ref.reload(BasicTestsF32::REF_ABS_F32_ID,mgr,nb);
        break;

        case BasicTestsF32::TEST_CLIP_F32_33:
          ref.reload(BasicTestsF32::REF_CLIP1_F32_ID,mgr);
          input1.reload(BasicTestsF32::INPUT_CLIP_F32_ID,mgr,ref.nbSamples());

          // Must be coherent with Python script used to generate test patterns
          this->min=-0.5f;
          this->max=-0.1f;
        break;

        case BasicTestsF32::TEST_CLIP_F32_34:
          ref.reload(BasicTestsF32::REF_CLIP2_F32_ID,mgr);
          input1.reload(BasicTestsF32::INPUT_CLIP_F32_ID,mgr,ref.nbSamples());
          // Must be coherent with Python script used to generate test patterns
          this->min=-0.5f;
          this->max=0.5f;
        break;

        case BasicTestsF32::TEST_CLIP_F32_35:
          ref.reload(BasicTestsF32::REF_CLIP3_F32_ID,mgr);
          input1.reload(BasicTestsF32::INPUT_CLIP_F32_ID,mgr,ref.nbSamples());
          // Must be coherent with Python script used to generate test patterns
          this->min=0.1f;
          this->max=0.5f;
        break;

       }
      

       if (id < TEST_CLIP_F32_33)
       {
         input1.reload(BasicTestsF32::INPUT1_F32_ID,mgr,nb);
         input2.reload(BasicTestsF32::INPUT2_F32_ID,mgr,nb);
       }

       output.create(ref.nbSamples(),BasicTestsF32::OUT_SAMPLES_F32_ID,mgr);
    }

    void BasicTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
