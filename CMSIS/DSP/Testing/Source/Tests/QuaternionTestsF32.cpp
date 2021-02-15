#include "QuaternionTestsF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)
#define ABS_ERROR (1.0e-7)



    void QuaternionTestsF32::test_quaternion_norm_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        arm_quaternion_norm_f32(inp1,outp,output.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void QuaternionTestsF32::test_quaternion_inverse_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        arm_quaternion_inverse_f32(inp1,outp,input1.nbSamples() >> 2);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void QuaternionTestsF32::test_quaternion_conjugate_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        arm_quaternion_conjugate_f32(inp1,outp,input1.nbSamples() >> 2);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void QuaternionTestsF32::test_quaternion_normalize_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        arm_quaternion_normalize_f32(inp1,outp,input1.nbSamples() >> 2);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void QuaternionTestsF32::test_quaternion_prod_single_f32()
    {
        const float32_t *inp1=input1.ptr();
        const float32_t *inp2=input2.ptr();
        float32_t *outp=output.ptr();

        for(uint32_t i=0; i < input1.nbSamples() >> 2; i++)
        {
           arm_quaternion_product_single_f32(inp1,inp2,outp);
           outp += 4;
           inp1 += 4;
           inp2 += 4;
        }

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void QuaternionTestsF32::test_quaternion_product_f32()
    {
        const float32_t *inp1=input1.ptr();
        const float32_t *inp2=input2.ptr();
        float32_t *outp=output.ptr();

        arm_quaternion_product_f32(inp1,inp2,outp,input1.nbSamples() >> 2);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void QuaternionTestsF32::test_quaternion2rotation_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        arm_quaternion2rotation_f32(inp1,outp,input1.nbSamples() >> 2);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void QuaternionTestsF32::test_rotation2quaternion_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        /*

        q and -q are representing the same rotation.
        To remove the ambiguity we force the real part ot be positive.
        Same convention followed in Python script.

        */

        arm_rotation2quaternion_f32(inp1,outp,output.nbSamples() >> 2);

        /*  Remove ambiguity */
        for(uint32_t i=0; i < output.nbSamples() >> 2 ; i++)
        {
            if (outp[0] < 0.0f)
            {
                outp[0] = -outp[0];
                outp[1] = -outp[1];
                outp[2] = -outp[2];
                outp[3] = -outp[3];
            }

            outp += 4;
        }

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 


 
    void QuaternionTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;

       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
          case QuaternionTestsF32::TEST_QUATERNION_NORM_F32_1:
            input1.reload(QuaternionTestsF32::INPUT1_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_NORM_F32_ID,mgr,nb);
          break;

          case QuaternionTestsF32::TEST_QUATERNION_INVERSE_F32_2:
            input1.reload(QuaternionTestsF32::INPUT1_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_INVERSE_F32_ID,mgr,nb);
          break;

          case QuaternionTestsF32::TEST_QUATERNION_CONJUGATE_F32_3:
            input1.reload(QuaternionTestsF32::INPUT1_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_CONJUGATE_F32_ID,mgr,nb);
          break;

          case QuaternionTestsF32::TEST_QUATERNION_NORMALIZE_F32_4:
            input1.reload(QuaternionTestsF32::INPUT1_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_NORMALIZE_F32_ID,mgr,nb);
          break;

          case QuaternionTestsF32::TEST_QUATERNION_PROD_SINGLE_F32_5:
            input1.reload(QuaternionTestsF32::INPUT1_F32_ID,mgr,nb);
            input2.reload(QuaternionTestsF32::INPUT2_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_MULT_F32_ID,mgr,nb);
          break;

          case QuaternionTestsF32::TEST_QUATERNION_PRODUCT_F32_6:
            input1.reload(QuaternionTestsF32::INPUT1_F32_ID,mgr,nb);
            input2.reload(QuaternionTestsF32::INPUT2_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_MULT_F32_ID,mgr,nb);
          break;

          case QuaternionTestsF32::TEST_QUATERNION2ROTATION_F32_7:
            input1.reload(QuaternionTestsF32::INPUT1_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_QUAT2ROT_F32_ID,mgr,nb);
          break;

          case QuaternionTestsF32::TEST_ROTATION2QUATERNION_F32_8:
            input1.reload(QuaternionTestsF32::INPUT7_F32_ID,mgr,nb);
            ref.reload(QuaternionTestsF32::REF_ROT2QUAT_F32_ID,mgr,nb);
          break;

       }
      

       

       output.create(ref.nbSamples(),QuaternionTestsF32::OUT_SAMPLES_F32_ID,mgr);
    }

    void QuaternionTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
