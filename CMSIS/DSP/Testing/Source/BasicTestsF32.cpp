#include "BasicTestsF32.h"
#include "Error.h"

#define FULL 1

#define GET_F32_PTR() \
const float32_t *inp1=input1.ptr(); \
const float32_t *inp2=input2.ptr(); \
float32_t *refp=ref.ptr(); \
float32_t *outp=output.ptr();

    void BasicTestsF32::test_add_f32()
    {
        GET_F32_PTR();

        arm_add_f32(inp1,inp2,outp,input1.nbSamples());
        

        ASSERT_NEAR_EQ(ref,output,(float)1e-6);

    } 
#ifdef FULL
    void BasicTestsF32::test_sub_f32()
    {
        GET_F32_PTR();

        arm_sub_f32(inp1,inp2,outp,input1.nbSamples());
        
        ASSERT_NEAR_EQ(ref,output,(float)1e-6);
       
    } 

    void BasicTestsF32::test_mult_f32()
    {
        GET_F32_PTR();

        arm_mult_f32(inp1,inp2,outp,input1.nbSamples());

        ASSERT_NEAR_EQ(ref,output,(float)1e-6);
       
    } 

    void BasicTestsF32::test_negate_f32()
    {
        GET_F32_PTR();

        arm_negate_f32(inp1,outp,input1.nbSamples());

        ASSERT_NEAR_EQ(ref,output,(float)1e-6);
       
    } 

    void BasicTestsF32::test_offset_f32()
    {
        GET_F32_PTR();

        arm_offset_f32(inp1,0.5,outp,input1.nbSamples());

        ASSERT_NEAR_EQ(ref,output,(float)1e-6);
       
    } 

    void BasicTestsF32::test_scale_f32()
    {
        GET_F32_PTR();

        arm_scale_f32(inp1,0.5,outp,input1.nbSamples());

        ASSERT_NEAR_EQ(ref,output,(float)1e-6);
       
    } 

    void BasicTestsF32::test_dot_prod_f32()
    {
        float32_t r;

        GET_F32_PTR();

        arm_dot_prod_f32(inp1,inp2,input1.nbSamples(),&r);

        ASSERT_NEAR_EQ(r,refp[0],(float)1e-6);
       
    } 

    void BasicTestsF32::test_abs_f32()
    {
        GET_F32_PTR();

        arm_abs_f32(inp1,outp,input1.nbSamples());

        ASSERT_NEAR_EQ(ref,output,(float)1e-6);
       
    } 

  #endif
    void BasicTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case BasicTestsF32::TEST_ADD_F32_1:
          nb = 3;
          ref.reload(BasicTestsF32::REF_ADD_F32_ID,mgr,nb);
          break;
  #ifdef FULL
        case BasicTestsF32::TEST_ADD_F32_2:
          nb = 8;
          ref.reload(BasicTestsF32::REF_ADD_F32_ID,mgr,nb);
          break;
        case BasicTestsF32::TEST_ADD_F32_3:
          nb = 9;
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
          nb = 9;
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
          nb = 9;
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
          nb = 9;
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
          nb = 9;
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
          nb = 9;
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
          nb = 9;
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
          nb = 9;
          ref.reload(BasicTestsF32::REF_ABS_F32_ID,mgr,nb);
          break;
#endif
       }
      

       input1.reload(BasicTestsF32::INPUT1_F32_ID,mgr,nb);
       input2.reload(BasicTestsF32::INPUT2_F32_ID,mgr,nb);

       output.create(input1.nbSamples(),BasicTestsF32::OUT_SAMPLES_F32_ID,mgr);
    }

    void BasicTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
