#include "BasicTestsF64.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 250

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (2.0e-13)

#define GET_F64_PTR() \
const float64_t *inp1=input1.ptr(); \
const float64_t *inp2=input2.ptr(); \
float64_t *outp=output.ptr();

    void BasicTestsF64::test_add_f64()
    {
        GET_F64_PTR();

        arm_add_f64(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

/*
    void BasicTestsF64::test_clip_f64()
    {
        const float64_t *inp=input1.ptr();
        float64_t *outp=output.ptr();

        arm_clip_f64(inp,outp,this->min, this->max,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 
*/
    void BasicTestsF64::test_sub_f64()
    {
        GET_F64_PTR();

        arm_sub_f64(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);
        
        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF64::test_mult_f64()
    {
        GET_F64_PTR();

        arm_mult_f64(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF64::test_negate_f64()
    {
        GET_F64_PTR();

        (void)inp2;

        arm_negate_f64(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF64::test_offset_f64()
    {
        GET_F64_PTR();

        (void)inp2;

        arm_offset_f64(inp1,0.5,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF64::test_scale_f64()
    {
        GET_F64_PTR();

        (void)inp2;

        arm_scale_f64(inp1,0.5,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

    void BasicTestsF64::test_dot_prod_f64()
    {
        float64_t r;

        GET_F64_PTR();

        arm_dot_prod_f64(inp1,inp2,input1.nbSamples(),&r);

        outp[0] = r;

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

        ASSERT_EMPTY_TAIL(output);

       
    } 

    void BasicTestsF64::test_abs_f64()
    {
        GET_F64_PTR();

        (void)inp2;

        arm_abs_f64(inp1,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
       
    } 

 
    void BasicTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;

       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case BasicTestsF64::TEST_ADD_F64_1:
          nb = 2;
          ref.reload(BasicTestsF64::REF_ADD_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_ADD_F64_2:
          nb = 4;
          ref.reload(BasicTestsF64::REF_ADD_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_ADD_F64_3:
          nb = 5;
          ref.reload(BasicTestsF64::REF_ADD_F64_ID,mgr,nb);
          break;


        case BasicTestsF64::TEST_SUB_F64_4:
          nb = 2;
          ref.reload(BasicTestsF64::REF_SUB_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_SUB_F64_5:
          nb = 4;
          ref.reload(BasicTestsF64::REF_SUB_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_SUB_F64_6:
          nb = 5;
          ref.reload(BasicTestsF64::REF_SUB_F64_ID,mgr,nb);
          break;

        case BasicTestsF64::TEST_MULT_F64_7:
          nb = 2;
          ref.reload(BasicTestsF64::REF_MULT_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_MULT_F64_8:
          nb = 4;
          ref.reload(BasicTestsF64::REF_MULT_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_MULT_F64_9:
          nb = 5;
          ref.reload(BasicTestsF64::REF_MULT_F64_ID,mgr,nb);
          break;

        case BasicTestsF64::TEST_NEGATE_F64_10:
          nb = 2;
          ref.reload(BasicTestsF64::REF_NEGATE_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_NEGATE_F64_11:
          nb = 4;
          ref.reload(BasicTestsF64::REF_NEGATE_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_NEGATE_F64_12:
          nb = 5;
          ref.reload(BasicTestsF64::REF_NEGATE_F64_ID,mgr,nb);
          break;

        case BasicTestsF64::TEST_OFFSET_F64_13:
          nb = 2;
          ref.reload(BasicTestsF64::REF_OFFSET_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_OFFSET_F64_14:
          nb = 4;
          ref.reload(BasicTestsF64::REF_OFFSET_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_OFFSET_F64_15:
          nb = 5;
          ref.reload(BasicTestsF64::REF_OFFSET_F64_ID,mgr,nb);
          break;

        case BasicTestsF64::TEST_SCALE_F64_16:
          nb = 2;
          ref.reload(BasicTestsF64::REF_SCALE_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_SCALE_F64_17:
          nb = 4;
          ref.reload(BasicTestsF64::REF_SCALE_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_SCALE_F64_18:
          nb = 5;
          ref.reload(BasicTestsF64::REF_SCALE_F64_ID,mgr,nb);
          break;

        case BasicTestsF64::TEST_DOT_PROD_F64_19:
          nb = 2;
          ref.reload(BasicTestsF64::REF_DOT_3_F64_ID,mgr);
          break;
        case BasicTestsF64::TEST_DOT_PROD_F64_20:
          nb = 4;
          ref.reload(BasicTestsF64::REF_DOT_4N_F64_ID,mgr);
          break;
        case BasicTestsF64::TEST_DOT_PROD_F64_21:
          nb = 5;
          ref.reload(BasicTestsF64::REF_DOT_4N1_F64_ID,mgr);
          break;

        case BasicTestsF64::TEST_ABS_F64_22:
          nb = 2;
          ref.reload(BasicTestsF64::REF_ABS_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_ABS_F64_23:
          nb = 4;
          ref.reload(BasicTestsF64::REF_ABS_F64_ID,mgr,nb);
          break;
        case BasicTestsF64::TEST_ABS_F64_24:
          nb = 5;
          ref.reload(BasicTestsF64::REF_ABS_F64_ID,mgr,nb);
          break;

        case BasicTestsF64::TEST_ADD_F64_25:
          ref.reload(BasicTestsF64::REF_ADD_F64_ID,mgr,nb);
        break;

        case BasicTestsF64::TEST_SUB_F64_26:
          ref.reload(BasicTestsF64::REF_SUB_F64_ID,mgr,nb);
        break;

        case BasicTestsF64::TEST_MULT_F64_27:
          ref.reload(BasicTestsF64::REF_MULT_F64_ID,mgr,nb);
        break;

        case BasicTestsF64::TEST_NEGATE_F64_28:
          ref.reload(BasicTestsF64::REF_NEGATE_F64_ID,mgr,nb);
        break;

        case BasicTestsF64::TEST_OFFSET_F64_29:
          ref.reload(BasicTestsF64::REF_OFFSET_F64_ID,mgr,nb);
        break;

        case BasicTestsF64::TEST_SCALE_F64_30:
          ref.reload(BasicTestsF64::REF_SCALE_F64_ID,mgr,nb);
        break;

        case BasicTestsF64::TEST_DOT_PROD_F64_31:
          ref.reload(BasicTestsF64::REF_DOT_LONG_F64_ID,mgr);
        break;

        case BasicTestsF64::TEST_ABS_F64_32:
          ref.reload(BasicTestsF64::REF_ABS_F64_ID,mgr,nb);
        break;

        case BasicTestsF64::TEST_CLIP_F64_33:
          ref.reload(BasicTestsF64::REF_CLIP1_F64_ID,mgr);
          input1.reload(BasicTestsF64::INPUT_CLIP_F64_ID,mgr,ref.nbSamples());

          // Must be coherent with Python script used to generate test patterns
          this->min=-0.5;
          this->max=-0.1;
        break;

        case BasicTestsF64::TEST_CLIP_F64_34:
          ref.reload(BasicTestsF64::REF_CLIP2_F64_ID,mgr);
          input1.reload(BasicTestsF64::INPUT_CLIP_F64_ID,mgr,ref.nbSamples());
          // Must be coherent with Python script used to generate test patterns
          this->min=-0.5;
          this->max=0.5;
        break;

        case BasicTestsF64::TEST_CLIP_F64_35:
          ref.reload(BasicTestsF64::REF_CLIP3_F64_ID,mgr);
          input1.reload(BasicTestsF64::INPUT_CLIP_F64_ID,mgr,ref.nbSamples());
          // Must be coherent with Python script used to generate test patterns
          this->min=0.1;
          this->max=0.5;
        break;

       }
      

       if (id < TEST_CLIP_F64_33)
       {
         input1.reload(BasicTestsF64::INPUT1_F64_ID,mgr,nb);
         input2.reload(BasicTestsF64::INPUT2_F64_ID,mgr,nb);
       }

       output.create(ref.nbSamples(),BasicTestsF64::OUT_SAMPLES_F64_ID,mgr);
    }

    void BasicTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
