#include "SupportTestsF16.h"
#include <stdlib.h>
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 120
#define REL_ERROR (1.0e-5)

#define ABS_WEIGHTEDSUM_ERROR (1.0e-1)
#define REL_WEIGHTEDSUM_ERROR (5.0e-3)

#define ABS_ERROR_F32 (1.0e-3)
#define REL_ERROR_F32 (1.0e-3)

#define ABS_Q15_ERROR ((q15_t)10)
#define ABS_Q31_ERROR ((q31_t)80)
#define ABS_Q7_ERROR ((q7_t)10)


void SupportTestsF16::test_weighted_sum_f16()
{
 const float16_t *inp = input.ptr();
 const float16_t *coefsp = coefs.ptr();
 float16_t *refp = ref.ptr();

 float16_t *outp = output.ptr();
 
 
 *outp=arm_weighted_sum_f16(inp, coefsp,this->nbSamples);

 ASSERT_CLOSE_ERROR(*outp,refp[this->offset],ABS_WEIGHTEDSUM_ERROR,REL_WEIGHTEDSUM_ERROR);
 ASSERT_EMPTY_TAIL(output);

} 


void SupportTestsF16::test_copy_f16()
{
 const float16_t *inp = input.ptr();
 float16_t *outp = output.ptr();
 
 
 arm_copy_f16(inp, outp,this->nbSamples);
 
 
 ASSERT_EQ(input,output);
 ASSERT_EMPTY_TAIL(output);

} 

void SupportTestsF16::test_fill_f16()
{
 float16_t *outp = output.ptr();
 float16_t val = 1.1;
 int i;
 

 arm_fill_f16(val, outp,this->nbSamples);
 
 
 for(i=0 ; i < this->nbSamples; i++)
 {
  ASSERT_EQ(val,outp[i]);
}
ASSERT_EMPTY_TAIL(output);

} 

void SupportTestsF16::test_f16_q15()
{
 const float16_t *inp = input.ptr();
 q15_t *outp = outputQ15.ptr();
 
 
 arm_f16_to_q15(inp, outp,this->nbSamples);
 
 
 ASSERT_NEAR_EQ(refQ15,outputQ15,ABS_Q15_ERROR);
 ASSERT_EMPTY_TAIL(outputQ15);

} 

void SupportTestsF16::test_f16_f32()
{
 const float16_t *inp = input.ptr();
 float32_t *outp = outputF32.ptr();
 
 
 arm_f16_to_float(inp, outp,this->nbSamples);
 
 
 ASSERT_REL_ERROR(refF32,outputF32,REL_ERROR_F32);
 ASSERT_EMPTY_TAIL(outputF32);

} 

void SupportTestsF16::test_q15_f16()
{
 const q15_t *inp = inputQ15.ptr();
 float16_t *outp = output.ptr();
 
 
 arm_q15_to_f16(inp, outp,this->nbSamples);
 
 
 ASSERT_REL_ERROR(ref,output,REL_ERROR);
 ASSERT_EMPTY_TAIL(outputF32);

} 

void SupportTestsF16::test_f32_f16()
{
 const float32_t *inp = inputF32.ptr();
 float16_t *outp = output.ptr();
 
 
 arm_float_to_f16(inp, outp,this->nbSamples);
 
 
 ASSERT_REL_ERROR(ref,output,REL_ERROR);
 ASSERT_EMPTY_TAIL(outputF32);

} 


void SupportTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
{

  (void)paramsArgs;
  switch(id)
  {    

    case TEST_WEIGHTED_SUM_F16_1:
    this->nbSamples = 7;
    input.reload(SupportTestsF16::INPUTS_F16_ID,mgr,this->nbSamples);
    coefs.reload(SupportTestsF16::WEIGHTS_F16_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::REF_F16_ID,mgr);

    output.create(1,SupportTestsF16::OUT_F16_ID,mgr);

    this->offset=0;
    break;

    case TEST_WEIGHTED_SUM_F16_2:
    this->nbSamples = 16;
    input.reload(SupportTestsF16::INPUTS_F16_ID,mgr,this->nbSamples);
    coefs.reload(SupportTestsF16::WEIGHTS_F16_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::REF_F16_ID,mgr);

    output.create(1,SupportTestsF16::OUT_F16_ID,mgr);

    this->offset=1;
    break;

    case TEST_WEIGHTED_SUM_F16_3:
    this->nbSamples = 23;
    input.reload(SupportTestsF16::INPUTS_F16_ID,mgr,this->nbSamples);
    coefs.reload(SupportTestsF16::WEIGHTS_F16_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::REF_F16_ID,mgr);

    output.create(1,SupportTestsF16::OUT_F16_ID,mgr);

    this->offset=2;
    break;

    case TEST_COPY_F16_4:
    this->nbSamples = 7;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);

    output.create(input.nbSamples(),SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_COPY_F16_5:
    this->nbSamples = 16;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);

    output.create(input.nbSamples(),SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_COPY_F16_6:
    this->nbSamples = 23;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);

    output.create(input.nbSamples(),SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_FILL_F16_7:
    this->nbSamples = 7;

    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_FILL_F16_8:
    this->nbSamples = 16;

    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_FILL_F16_9:
    this->nbSamples = 23;

    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_F16_Q15_10:
    this->nbSamples = 7;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    refQ15.reload(SupportTestsF16::SAMPLES_Q15_ID,mgr,this->nbSamples);
    outputQ15.create(this->nbSamples,SupportTestsF16::OUT_Q15_ID,mgr);

    break;

    case TEST_F16_Q15_11:
    this->nbSamples = 16;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    refQ15.reload(SupportTestsF16::SAMPLES_Q15_ID,mgr,this->nbSamples);
    outputQ15.create(this->nbSamples,SupportTestsF16::OUT_Q15_ID,mgr);

    break;

    case TEST_F16_Q15_12:
    this->nbSamples = 23;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    refQ15.reload(SupportTestsF16::SAMPLES_Q15_ID,mgr,this->nbSamples);
    outputQ15.create(this->nbSamples,SupportTestsF16::OUT_Q15_ID,mgr);

    break;

    case TEST_F16_F32_13:
    this->nbSamples = 7;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    refF32.reload(SupportTestsF16::SAMPLES_F32_ID,mgr,this->nbSamples);
    outputF32.create(this->nbSamples,SupportTestsF16::OUT_F32_ID,mgr);

    break;

    case TEST_F16_F32_14:
    this->nbSamples = 16;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    refF32.reload(SupportTestsF16::SAMPLES_F32_ID,mgr,this->nbSamples);
    outputF32.create(this->nbSamples,SupportTestsF16::OUT_F32_ID,mgr);

    break;

    case TEST_F16_F32_15:
    this->nbSamples = 23;
    input.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    refF32.reload(SupportTestsF16::SAMPLES_F32_ID,mgr,this->nbSamples);
    outputF32.create(this->nbSamples,SupportTestsF16::OUT_F32_ID,mgr);

    break;

    case TEST_Q15_F16_16:
    this->nbSamples = 7;
    inputQ15.reload(SupportTestsF16::SAMPLES_Q15_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_Q15_F16_17:
    this->nbSamples = 16;
    inputQ15.reload(SupportTestsF16::SAMPLES_Q15_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_Q15_F16_18:
    this->nbSamples = 23;
    inputQ15.reload(SupportTestsF16::SAMPLES_Q15_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_F32_F16_19:
    this->nbSamples = 7;
    inputF32.reload(SupportTestsF16::SAMPLES_F32_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_F32_F16_20:
    this->nbSamples = 16;
    inputF32.reload(SupportTestsF16::SAMPLES_F32_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;

    case TEST_F32_F16_21:
    this->nbSamples = 23;
    inputF32.reload(SupportTestsF16::SAMPLES_F32_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF16::SAMPLES_F16_ID,mgr,this->nbSamples);
    output.create(this->nbSamples,SupportTestsF16::OUT_F16_ID,mgr);

    break;


  }       

}

void SupportTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
{
  (void)id;
  output.dump(mgr);
}
