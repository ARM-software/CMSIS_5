#include "GoertzelTestsF32.h"
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#include <cstdio>

#define SNR_THRESHOLD 120
#define REL_ERROR (1.0e-5)
#define ABS_Q15_ERROR ((q15_t)10)
#define ABS_Q31_ERROR ((q31_t)80)
#define ABS_Q7_ERROR ((q7_t)10)

    void GoertzelTestsF32::test_goertzel_const_f32()
    {
       float32_t *inp = input.ptr();
       float32_t *outp = output.ptr();
       float32_t *buf = buffer.ptr();
       arm_goertzel_instance_f32 S;
       buf = (float32_t*)malloc(8*sizeof(float32_t));

       arm_goertzel_init_f32( &S, 0, buf, 1);
       arm_goertzel_f32( &S, inp, outp, 20);
        
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref, (float32_t)SNR_THRESHOLD);
    } 

    void GoertzelTestsF32::test_goertzel_sin_f32()
    {
       float32_t *inp = input.ptr();
       float32_t *outp = output.ptr();
       float32_t pi = PI;
       float32_t *buf = buffer.ptr();
       arm_goertzel_instance_f32 S;
       buf = (float32_t*)malloc(8*sizeof(float32_t));

       arm_goertzel_init_f32( &S, (2*PI*2*0.0625f), buf, 1); // 2/16
       arm_goertzel_f32( &S, inp, outp, 16);
        
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref, (float32_t)SNR_THRESHOLD);
    } 

    void GoertzelTestsF32::test_goertzel_rand_f32()
    {
       float32_t *inp = input.ptr();
       float32_t *outp = output.ptr();
       float32_t pi = PI;
       float32_t *buf = buffer.ptr();
       arm_goertzel_instance_f32 S;
       buf = (float32_t*)malloc(8*sizeof(float32_t));

       arm_goertzel_init_f32( &S, (2*PI*0.25f), buf, 1); // 10/40
       arm_goertzel_f32( &S, inp, outp, 40);
        
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref, (float32_t)SNR_THRESHOLD);
    } 


    void GoertzelTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        switch(id)
        {    
        case TEST_GOERTZEL_CONST_F32_1:
              input.reload(GoertzelTestsF32::INPUT_GOERTZEL_CONST_F32_ID,mgr,20);
              ref.reload(GoertzelTestsF32::REF_GOERTZEL_CONST_F32_ID,mgr,2);
              output.create(2,GoertzelTestsF32::OUT_F32_ID,mgr);            
        break;

        case TEST_GOERTZEL_SIN_F32_2:
              input.reload(GoertzelTestsF32::INPUT_GOERTZEL_SIN_F32_ID,mgr,16);
              ref.reload(GoertzelTestsF32::REF_GOERTZEL_SIN_F32_ID,mgr,2);
              output.create(2,GoertzelTestsF32::OUT_F32_ID,mgr);            
        break;

        case TEST_GOERTZEL_RAND_F32_3:
              input.reload(GoertzelTestsF32::INPUT_GOERTZEL_RAND_F32_ID,mgr,40);
              ref.reload(GoertzelTestsF32::REF_GOERTZEL_RAND_F32_ID,mgr,2);
              output.create(2,GoertzelTestsF32::OUT_F32_ID,mgr);            
        break;

        }       
    }

    void GoertzelTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
