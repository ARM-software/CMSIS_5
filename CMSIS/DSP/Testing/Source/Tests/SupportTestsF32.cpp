#include "SupportTestsF32.h"
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#include <cstdio>

#define SNR_THRESHOLD 120
#define REL_ERROR (1.0e-5)
#define ABS_Q15_ERROR ((q15_t)10)
#define ABS_Q31_ERROR ((q31_t)80)
#define ABS_Q7_ERROR ((q7_t)10)


    void SupportTestsF32::test_weighted_sum_f32()
    {
       const float32_t *inp = input.ptr();
       const float32_t *coefsp = coefs.ptr();
       float32_t *refp = ref.ptr();

       float32_t *outp = output.ptr();
       
      
       *outp=arm_weighted_sum_f32(inp, coefsp,this->nbSamples);
         
          
        ASSERT_REL_ERROR(*outp,refp[this->offset],REL_ERROR);
        ASSERT_EMPTY_TAIL(output);

    } 

    void SupportTestsF32::test_copy_f32()
    {
       const float32_t *inp = input.ptr();
       float32_t *outp = output.ptr();
       
      
       arm_copy_f32(inp, outp,this->nbSamples);
         
          
       ASSERT_EQ(input,output);
       ASSERT_EMPTY_TAIL(output);

    } 

    void SupportTestsF32::test_fill_f32()
    {
       float32_t *outp = output.ptr();
       float32_t val = 1.1;
       int i;
      

       arm_fill_f32(val, outp,this->nbSamples);
         
          
       for(i=0 ; i < this->nbSamples; i++)
       {
          ASSERT_EQ(val,outp[i]);
       }
       ASSERT_EMPTY_TAIL(output);

    } 

    void SupportTestsF32::test_float_to_q15()
    {
       const float32_t *inp = input.ptr();
       q15_t *refp = refQ15.ptr();
       q15_t *outp = outputQ15.ptr();
       
      
       arm_float_to_q15(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ15,outputQ15,ABS_Q15_ERROR);
       ASSERT_EMPTY_TAIL(outputQ15);

    } 

    void SupportTestsF32::test_float_to_q31()
    {
       const float32_t *inp = input.ptr();
       q31_t *refp = refQ31.ptr();
       q31_t *outp = outputQ31.ptr();
       
      
       arm_float_to_q31(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ31,outputQ31,ABS_Q31_ERROR);
       ASSERT_EMPTY_TAIL(outputQ31);

    } 

    void SupportTestsF32::test_float_to_q7()
    {
       const float32_t *inp = input.ptr();
       q7_t *refp = refQ7.ptr();
       q7_t *outp = outputQ7.ptr();
       
      
       arm_float_to_q7(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ7,outputQ7,ABS_Q7_ERROR);
       ASSERT_EMPTY_TAIL(outputQ7);

    } 

  
    void SupportTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        switch(id)
        {
           
            case TEST_WEIGHTED_SUM_F32_1:
              this->nbSamples = 3;
              input.reload(SupportTestsF32::INPUTS_F32_ID,mgr,this->nbSamples);
              coefs.reload(SupportTestsF32::WEIGHTS_F32_ID,mgr,this->nbSamples);
              ref.reload(SupportTestsF32::REF_F32_ID,mgr);

              output.create(1,SupportTestsF32::OUT_F32_ID,mgr);

              this->offset=0;
            break;

            case TEST_WEIGHTED_SUM_F32_2:
              this->nbSamples = 8;
              input.reload(SupportTestsF32::INPUTS_F32_ID,mgr,this->nbSamples);
              coefs.reload(SupportTestsF32::WEIGHTS_F32_ID,mgr,this->nbSamples);
              ref.reload(SupportTestsF32::REF_F32_ID,mgr);

              output.create(1,SupportTestsF32::OUT_F32_ID,mgr);

              this->offset=1;
            break;

            case TEST_WEIGHTED_SUM_F32_3:
              this->nbSamples = 9;
              input.reload(SupportTestsF32::INPUTS_F32_ID,mgr,this->nbSamples);
              coefs.reload(SupportTestsF32::WEIGHTS_F32_ID,mgr,this->nbSamples);
              ref.reload(SupportTestsF32::REF_F32_ID,mgr);

              output.create(1,SupportTestsF32::OUT_F32_ID,mgr);

              this->offset=2;
            break;

            case TEST_COPY_F32_4:
              this->nbSamples = 3;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);

              output.create(input.nbSamples(),SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_COPY_F32_5:
              this->nbSamples = 8;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);

              output.create(input.nbSamples(),SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_COPY_F32_6:
              this->nbSamples = 9;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);

              output.create(input.nbSamples(),SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FILL_F32_7:
              this->nbSamples = 3;

              output.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FILL_F32_8:
              this->nbSamples = 8;

              output.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FILL_F32_9:
              this->nbSamples = 9;

              output.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q15_10:
              this->nbSamples = 7;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsF32::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q15_11:
              this->nbSamples = 16;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsF32::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q15_12:
              this->nbSamples = 17;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsF32::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q31_13:
              this->nbSamples = 3;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsF32::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q31_14:
              this->nbSamples = 8;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsF32::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q31_15:
              this->nbSamples = 9;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsF32::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q7_16:
              this->nbSamples = 15;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsF32::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q7_17:
              this->nbSamples = 32;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsF32::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

            case TEST_FLOAT_TO_Q7_18:
              this->nbSamples = 33;
              input.reload(SupportTestsF32::SAMPLES_F32_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsF32::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsF32::OUT_F32_ID,mgr);

            break;

        }

       

    }

    void SupportTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
