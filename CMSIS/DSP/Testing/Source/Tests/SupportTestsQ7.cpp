#include "SupportTestsQ7.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#define SNR_THRESHOLD 120
#define REL_ERROR (1.0e-5)
#define ABS_Q15_ERROR ((q15_t)(1<<8))
#define ABS_Q31_ERROR ((q31_t)(1<<24))
#define ABS_Q7_ERROR ((q7_t)10)

#if defined ( __CC_ARM )
#pragma diag_suppress 170
#endif

    void SupportTestsQ7::test_copy_q7()
    {
       const q7_t *inp = inputQ7.ptr();
       q7_t *outp = outputQ7.ptr();
       
      
       arm_copy_q7(inp, outp,this->nbSamples);
         
          
       ASSERT_EQ(inputQ7,outputQ7);
       ASSERT_EMPTY_TAIL(outputQ7);

    } 

    void SupportTestsQ7::test_fill_q7()
    {
       q7_t *outp = outputQ7.ptr();
       q7_t val = 0x40;
       int i;
      

       arm_fill_q7(val, outp,this->nbSamples);
         
          
       for(i=0 ; i < this->nbSamples; i++)
       {
          ASSERT_EQ(val,outp[i]);
       }

       ASSERT_EMPTY_TAIL(outputQ7);

    } 

    void SupportTestsQ7::test_q7_float()
    {
       const q7_t *inp = inputQ7.ptr();
       float32_t *refp = refF32.ptr();
       float32_t *outp = outputF32.ptr();
       
      
       arm_q7_to_float(inp, outp,this->nbSamples);
         
          
       ASSERT_CLOSE_ERROR(refF32,outputF32,0.01,REL_ERROR);

       ASSERT_EMPTY_TAIL(outputF32);

    } 

    void SupportTestsQ7::test_q7_q31()
    {
       const q7_t *inp = inputQ7.ptr();
       q31_t *refp = refQ31.ptr();
       q31_t *outp = outputQ31.ptr();
       
      
       arm_q7_to_q31(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ31,outputQ31,ABS_Q31_ERROR);
       ASSERT_EMPTY_TAIL(outputQ31);

    } 

    void SupportTestsQ7::test_q7_q15()
    {
       const q7_t *inp = inputQ7.ptr();
       q15_t *refp = refQ15.ptr();
       q15_t *outp = outputQ15.ptr();
       
      
       arm_q7_to_q15(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ15,outputQ15,ABS_Q15_ERROR);
       ASSERT_EMPTY_TAIL(outputQ15);

    } 

    static const q7_t testReadQ7[4]={-4,-3,-2,1};
    static q7_t testWriteQ7[4]={0,0,0,0};

    void SupportTestsQ7::test_read_q7x4_ia()
    {
        q31_t result=0;
        q7_t *p = (q7_t*)testReadQ7;

        result = read_q7x4_ia(&p);
        printf("%08X\n",result);

        ASSERT_TRUE(result == 0x01FEFDFC);
        ASSERT_TRUE(p == testReadQ7 + 4);
    }

   void SupportTestsQ7::test_read_q7x4_da()
    {

        q31_t result=0;
        q7_t *p = (q7_t*)testReadQ7;

        result = read_q7x4_da(&p);

        ASSERT_TRUE(result == 0x01FEFDFC);
        ASSERT_TRUE(p == testReadQ7 - 4);
    }

    void SupportTestsQ7::test_write_q7x4_ia()
    {
        q31_t val = 0x01FEFDFC;
        q7_t *p = (q7_t*)testWriteQ7;

        testWriteQ7[0] = 0;
        testWriteQ7[1] = 0;
        testWriteQ7[2] = 0;
        testWriteQ7[3] = 0;

        write_q7x4_ia(&p,val);

        ASSERT_TRUE(testWriteQ7[0] == -4);
        ASSERT_TRUE(testWriteQ7[1] == -3);
        ASSERT_TRUE(testWriteQ7[2] == -2);
        ASSERT_TRUE(testWriteQ7[3] == 1);
        ASSERT_TRUE(p == testWriteQ7 + 4);

    }

  
    void SupportTestsQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        switch(id)
        {
 
            case TEST_COPY_Q7_1:
              this->nbSamples = 15;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);

              outputQ7.create(inputQ7.nbSamples(),SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_COPY_Q7_2:
              this->nbSamples = 32;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);

              outputQ7.create(inputQ7.nbSamples(),SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_COPY_Q7_3:
              this->nbSamples = 47;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);

              outputQ7.create(inputQ7.nbSamples(),SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_FILL_Q7_4:
              this->nbSamples = 15;

              outputQ7.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_FILL_Q7_5:
              this->nbSamples = 32;

              outputQ7.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_FILL_Q7_6:
              this->nbSamples = 47;

              outputQ7.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_FLOAT_7:
              this->nbSamples = 15;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ7::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_FLOAT_8:
              this->nbSamples = 32;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ7::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_FLOAT_9:
              this->nbSamples = 47;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ7::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_Q31_10:
              this->nbSamples = 15;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsQ7::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_Q31_11:
              this->nbSamples = 32;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsQ7::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_Q31_12:
              this->nbSamples = 47;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsQ7::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_Q15_13:
              this->nbSamples = 15;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsQ7::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_Q15_14:
              this->nbSamples = 32;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsQ7::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

            case TEST_Q7_Q15_15:
              this->nbSamples = 47;
              inputQ7.reload(SupportTestsQ7::SAMPLES_Q7_ID,mgr,this->nbSamples);
              refQ15.reload(SupportTestsQ7::SAMPLES_Q15_ID,mgr,this->nbSamples);
              outputQ15.create(this->nbSamples,SupportTestsQ7::OUT_ID,mgr);

            break;

        }

       

    }

    void SupportTestsQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      switch(id)
      {
 
            case TEST_COPY_Q7_1:
            case TEST_COPY_Q7_2:
            case TEST_COPY_Q7_3:
            case TEST_FILL_Q7_4:
            case TEST_FILL_Q7_5:
            case TEST_FILL_Q7_6:
               outputQ7.dump(mgr);
            break;

            case TEST_Q7_FLOAT_7:
            case TEST_Q7_FLOAT_8:
            case TEST_Q7_FLOAT_9:
               outputF32.dump(mgr);
            break;

            case TEST_Q7_Q31_10:
            case TEST_Q7_Q31_11:
            case TEST_Q7_Q31_12:
               outputQ31.dump(mgr);
            break;

            case TEST_Q7_Q15_13:
            case TEST_Q7_Q15_14:
            case TEST_Q7_Q15_15:
               outputQ15.dump(mgr);
            break;
      }
    }
