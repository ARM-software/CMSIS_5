#include "SupportTestsQ31.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 120
#define REL_ERROR (1.0e-5)
#define ABS_Q15_ERROR ((q15_t)10)
#define ABS_Q31_ERROR ((q31_t)80)
#define ABS_Q7_ERROR ((q7_t)10)

    void SupportTestsQ31::test_copy_q31()
    {
       const q31_t *inp = inputQ31.ptr();
       q31_t *outp = outputQ31.ptr();
       
      
       arm_copy_q31(inp, outp,this->nbSamples);
         
          
       ASSERT_EQ(inputQ31,outputQ31);
       ASSERT_EMPTY_TAIL(outputQ31);

    } 

    void SupportTestsQ31::test_fill_q31()
    {
       q31_t *outp = outputQ31.ptr();
       q31_t val = 0x4000;
       int i;
      

       arm_fill_q31(val, outp,this->nbSamples);
         
          
       for(i=0 ; i < this->nbSamples; i++)
       {
          ASSERT_EQ(val,outp[i]);
       }
       ASSERT_EMPTY_TAIL(outputQ31);

    } 

    void SupportTestsQ31::test_q31_float()
    {
       const q31_t *inp = inputQ31.ptr();
       float32_t *outp = outputF32.ptr();
       
      
       arm_q31_to_float(inp, outp,this->nbSamples);
         
          
       ASSERT_REL_ERROR(refF32,outputF32,REL_ERROR);
       ASSERT_EMPTY_TAIL(outputF32);

    } 

    void SupportTestsQ31::test_q31_q15()
    {
       const q31_t *inp = inputQ31.ptr();
       q15_t *outp = outputQ15.ptr();
       
      
       arm_q31_to_q15(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ15,outputQ15,ABS_Q15_ERROR);
       ASSERT_EMPTY_TAIL(outputQ15);

    } 

    void SupportTestsQ31::test_q31_q7()
    {
       const q31_t *inp = inputQ31.ptr();
       q7_t *outp = outputQ7.ptr();
       
      
       arm_q31_to_q7(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ7,outputQ7,ABS_Q7_ERROR);
       ASSERT_EMPTY_TAIL(outputQ7);

    } 

  
    void SupportTestsQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        (void)paramsArgs;
        switch(id)
        {
 
            case TEST_COPY_Q31_1:
              this->nbSamples = 3;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);

              outputQ31.create(inputQ31.nbSamples(),SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_COPY_Q31_2:
              this->nbSamples = 8;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);

              outputQ31.create(inputQ31.nbSamples(),SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_COPY_Q31_3:
              this->nbSamples = 11;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);

              outputQ31.create(inputQ31.nbSamples(),SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_FILL_Q31_4:
              this->nbSamples = 3;

              outputQ31.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_FILL_Q31_5:
              this->nbSamples = 8;

              outputQ31.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_FILL_Q31_6:
              this->nbSamples = 11;

              outputQ31.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_FLOAT_7:
              this->nbSamples = 7;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ31::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_FLOAT_8:
              this->nbSamples = 16;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ31::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_FLOAT_9:
              this->nbSamples = 17;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ31::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_Q15_10:
              this->nbSamples = 3;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsQ31::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_Q15_11:
              this->nbSamples = 8;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsQ31::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_Q15_12:
              this->nbSamples = 11;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsQ31::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_Q7_13:
              this->nbSamples = 15;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsQ31::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_Q7_14:
              this->nbSamples = 32;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsQ31::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

            case TEST_Q31_Q7_15:
              this->nbSamples = 33;
              inputQ31.reload(SupportTestsQ31::SAMPLES_Q31_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsQ31::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsQ31::OUT_ID,mgr);

            break;

        }

       

    }

    void SupportTestsQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      switch(id)
      {
 
            case TEST_COPY_Q31_1:
            case TEST_COPY_Q31_2:
            case TEST_COPY_Q31_3:
            case TEST_FILL_Q31_4:
            case TEST_FILL_Q31_5:
            case TEST_FILL_Q31_6:
               outputQ31.dump(mgr);
            break;

            case TEST_Q31_FLOAT_7:
            case TEST_Q31_FLOAT_8:
            case TEST_Q31_FLOAT_9:
               outputF32.dump(mgr);
            break;

            case TEST_Q31_Q15_10:
            case TEST_Q31_Q15_11:
            case TEST_Q31_Q15_12:
               outputQ15.dump(mgr);
            break;

            case TEST_Q31_Q7_13:
            case TEST_Q31_Q7_14:
            case TEST_Q31_Q7_15:
               outputQ7.dump(mgr);
            break;
      }
    }
