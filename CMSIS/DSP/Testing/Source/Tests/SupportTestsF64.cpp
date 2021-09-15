#include "SupportTestsF64.h"
#include <stdlib.h>
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 120
#define REL_ERROR (1.0e-5)
#define ABS_Q15_ERROR ((q15_t)10)
#define ABS_Q31_ERROR ((q31_t)80)
#define ABS_Q7_ERROR ((q7_t)10)

/*
void SupportTestsF64::test_weighted_sum_f64()
{
 const float64_t *inp = input.ptr();
 const float64_t *coefsp = coefs.ptr();
 float64_t *refp = ref.ptr();

 float64_t *outp = output.ptr();
 
 
 *outp=arm_weighted_sum_f64(inp, coefsp,this->nbSamples);
 
 
 ASSERT_REL_ERROR(*outp,refp[this->offset],REL_ERROR);
 ASSERT_EMPTY_TAIL(output);

} 
*/
void SupportTestsF64::test_copy_f64()
{
 const float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 
 
 arm_copy_f64(inp, outp,this->nbSamples);
 
 
 ASSERT_EQ(input,output);
 ASSERT_EMPTY_TAIL(output);

} 

void SupportTestsF64::test_fill_f64()
{
 float64_t *outp = output.ptr();
 float64_t val = 1.1;
 int i;
 

 arm_fill_f64(val, outp,this->nbSamples);
 
 
 for(i=0 ; i < this->nbSamples; i++)
 {
  ASSERT_EQ(val,outp[i]);
}
ASSERT_EMPTY_TAIL(output);

} 

/*
void SupportTestsF64::test_float_to_q15()
{
 const float64_t *inp = input.ptr();
 q15_t *outp = outputQ15.ptr();
 
 
 arm_float_to_q15(inp, outp,this->nbSamples);
 
 
 ASSERT_NEAR_EQ(refQ15,outputQ15,ABS_Q15_ERROR);
 ASSERT_EMPTY_TAIL(outputQ15);

} 

void SupportTestsF64::test_float_to_q31()
{
 const float64_t *inp = input.ptr();
 q31_t *outp = outputQ31.ptr();
 
 
 arm_float_to_q31(inp, outp,this->nbSamples);
 
 
 ASSERT_NEAR_EQ(refQ31,outputQ31,ABS_Q31_ERROR);
 ASSERT_EMPTY_TAIL(outputQ31);

} 

void SupportTestsF64::test_float_to_q7()
{
 const float64_t *inp = input.ptr();
 q7_t *outp = outputQ7.ptr();
 
 
 arm_float_to_q7(inp, outp,this->nbSamples);
 
 
 ASSERT_NEAR_EQ(refQ7,outputQ7,ABS_Q7_ERROR);
 ASSERT_EMPTY_TAIL(outputQ7);

} 

void SupportTestsF64::test_bitonic_sort_out_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_BITONIC, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);

 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_bitonic_sort_in_f64()
{
 float64_t *inp = input.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_BITONIC, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,inp,this->nbSamples);

 ASSERT_EMPTY_TAIL(input);

 ASSERT_EQ(input,ref);

} 

void SupportTestsF64::test_bitonic_sort_const_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_BITONIC, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);

 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_bubble_sort_out_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_BUBBLE, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);

 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_bubble_sort_in_f64()
{
 float64_t *inp = input.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_BUBBLE, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,inp,this->nbSamples);

 ASSERT_EMPTY_TAIL(input);

 ASSERT_EQ(input,ref);

} 

void SupportTestsF64::test_bubble_sort_const_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_BUBBLE, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);

 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_heap_sort_out_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_HEAP, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_heap_sort_in_f64()
{
 float64_t *inp = input.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_HEAP, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,inp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(input);

 ASSERT_EQ(input,ref);
} 

void SupportTestsF64::test_heap_sort_const_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_HEAP, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_insertion_sort_out_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_INSERTION, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_insertion_sort_in_f64()
{
 float64_t *inp = input.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_INSERTION, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,inp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(input);

 ASSERT_EQ(input,ref);

} 

void SupportTestsF64::test_insertion_sort_const_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_INSERTION, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_merge_sort_out_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 float64_t *buf = buffer.ptr();
 buf = (float64_t *)malloc((this->nbSamples)*sizeof(float64_t) );
 arm_merge_sort_instance_f64 S;

 arm_merge_sort_init_f64(&S, ARM_SORT_ASCENDING, buf);
 arm_merge_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_merge_sort_const_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 float64_t *buf = buffer.ptr();
 buf = (float64_t *)malloc((this->nbSamples)*sizeof(float64_t) );
 arm_merge_sort_instance_f64 S;

 arm_merge_sort_init_f64(&S, ARM_SORT_ASCENDING, buf);
 arm_merge_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);
} 

void SupportTestsF64::test_quick_sort_out_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_QUICK, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_quick_sort_in_f64()
{
 float64_t *inp = input.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_QUICK, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,inp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(input);

 ASSERT_EQ(input,ref);

} 

void SupportTestsF64::test_quick_sort_const_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_QUICK, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_selection_sort_out_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_SELECTION, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

void SupportTestsF64::test_selection_sort_in_f64()
{
 float64_t *inp = input.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_SELECTION, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,inp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(input);

 ASSERT_EQ(input,ref);

} 

void SupportTestsF64::test_selection_sort_const_f64()
{
 float64_t *inp = input.ptr();
 float64_t *outp = output.ptr();
 arm_sort_instance_f64 S;

 arm_sort_init_f64(&S, ARM_SORT_SELECTION, ARM_SORT_ASCENDING);

 arm_sort_f64(&S,inp,outp,this->nbSamples);
 
 ASSERT_EMPTY_TAIL(output);

 ASSERT_EQ(output,ref);

} 

*/
void SupportTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
{

  (void)paramsArgs;
  switch(id)
  {    
    /*case TEST_WEIGHTED_SUM_F64_1:
    this->nbSamples = 2;
    input.reload(SupportTestsF64::INPUTS_F64_ID,mgr,this->nbSamples);
    coefs.reload(SupportTestsF64::WEIGHTS_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_F64_ID,mgr);

    output.create(1,SupportTestsF64::OUT_F64_ID,mgr);

    this->offset=0;
    break;

    case TEST_WEIGHTED_SUM_F64_2:
    this->nbSamples = 4;
    input.reload(SupportTestsF64::INPUTS_F64_ID,mgr,this->nbSamples);
    coefs.reload(SupportTestsF64::WEIGHTS_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_F64_ID,mgr);

    output.create(1,SupportTestsF64::OUT_F64_ID,mgr);

    this->offset=1;
    break;

    case TEST_WEIGHTED_SUM_F64_3:
    this->nbSamples = 5;
    input.reload(SupportTestsF64::INPUTS_F64_ID,mgr,this->nbSamples);
    coefs.reload(SupportTestsF64::WEIGHTS_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_F64_ID,mgr);

    output.create(1,SupportTestsF64::OUT_F64_ID,mgr);

    this->offset=2;
    break;
*/
    case TEST_COPY_F64_4:
    this->nbSamples = 2;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);

    output.create(input.nbSamples(),SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_COPY_F64_5:
    this->nbSamples = 4;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);

    output.create(input.nbSamples(),SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_COPY_F64_6:
    this->nbSamples = 5;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);

    output.create(input.nbSamples(),SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FILL_F64_7:
    this->nbSamples = 2;

    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FILL_F64_8:
    this->nbSamples = 4;

    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FILL_F64_9:
    this->nbSamples = 5;

    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;
/*
    case TEST_FLOAT_TO_Q15_10:
    this->nbSamples = 7;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ15.reload(SupportTestsF64::SAMPLES_Q15_ID,mgr,this->nbSamples);
    outputQ15.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q15_11:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ15.reload(SupportTestsF64::SAMPLES_Q15_ID,mgr,this->nbSamples);
    outputQ15.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q15_12:
    this->nbSamples = 17;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ15.reload(SupportTestsF64::SAMPLES_Q15_ID,mgr,this->nbSamples);
    outputQ15.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q31_13:
    this->nbSamples = 3;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ31.reload(SupportTestsF64::SAMPLES_Q31_ID,mgr,this->nbSamples);
    outputQ31.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q31_14:
    this->nbSamples = 8;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ31.reload(SupportTestsF64::SAMPLES_Q31_ID,mgr,this->nbSamples);
    outputQ31.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q31_15:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ31.reload(SupportTestsF64::SAMPLES_Q31_ID,mgr,this->nbSamples);
    outputQ31.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q7_16:
    this->nbSamples = 15;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ7.reload(SupportTestsF64::SAMPLES_Q7_ID,mgr,this->nbSamples);
    outputQ7.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q7_17:
    this->nbSamples = 32;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ7.reload(SupportTestsF64::SAMPLES_Q7_ID,mgr,this->nbSamples);
    outputQ7.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_FLOAT_TO_Q7_18:
    this->nbSamples = 33;
    input.reload(SupportTestsF64::SAMPLES_F64_ID,mgr,this->nbSamples);
    refQ7.reload(SupportTestsF64::SAMPLES_Q7_ID,mgr,this->nbSamples);
    outputQ7.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);

    break;

    case TEST_BITONIC_SORT_OUT_F64_19:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_BITONIC_SORT_16_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_BITONIC_SORT_16_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);
    break;

    case TEST_BITONIC_SORT_OUT_F64_20:
    this->nbSamples = 32;
    input.reload(SupportTestsF64::INPUT_BITONIC_SORT_64_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_BITONIC_SORT_64_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_BITONIC_SORT_IN_F64_21:
    this->nbSamples = 32;
    input.reload(SupportTestsF64::INPUT_BITONIC_SORT_64_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_BITONIC_SORT_64_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_BITONIC_SORT_CONST_F64_22:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_SORT_CONST_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_CONST_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_BUBBLE_SORT_OUT_F64_23:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_BUBBLE_SORT_IN_F64_24:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_BUBBLE_SORT_CONST_F64_25:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_SORT_CONST_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_CONST_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_HEAP_SORT_OUT_F64_26:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_HEAP_SORT_IN_F64_27:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_HEAP_SORT_CONST_F64_28:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_SORT_CONST_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_CONST_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_INSERTION_SORT_OUT_F64_29:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_INSERTION_SORT_IN_F64_30:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_INSERTION_SORT_CONST_F64_31:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_SORT_CONST_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_CONST_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_MERGE_SORT_OUT_F64_32:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_MERGE_SORT_CONST_F64_33:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_SORT_CONST_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_CONST_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_QUICK_SORT_OUT_F64_34:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_QUICK_SORT_IN_F64_35:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_QUICK_SORT_CONST_F64_36:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_SORT_CONST_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_CONST_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;

    case TEST_SELECTION_SORT_OUT_F64_37:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_SELECTION_SORT_IN_F64_38:
    this->nbSamples = 11;
    input.reload(SupportTestsF64::INPUT_SORT_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr);            
    break;

    case TEST_SELECTION_SORT_CONST_F64_39:
    this->nbSamples = 16;
    input.reload(SupportTestsF64::INPUT_SORT_CONST_F64_ID,mgr,this->nbSamples);
    ref.reload(SupportTestsF64::REF_SORT_CONST_F64_ID,mgr);
    output.create(this->nbSamples,SupportTestsF64::OUT_F64_ID,mgr); 
    break;
*/


  }       

}

void SupportTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
{
  (void)id;
  output.dump(mgr);
}
