#include "SupportTestsQ15.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#define SNR_THRESHOLD 120
#define REL_ERROR (1.0e-3)
#define ABS_Q15_ERROR ((q15_t)10)
#define ABS_Q31_ERROR ((q31_t)40000)
#define ABS_Q7_ERROR ((q7_t)10)

#if defined ( __CC_ARM )
#pragma diag_suppress 170
#endif

    void SupportTestsQ15::test_copy_q15()
    {
       const q15_t *inp = inputQ15.ptr();
       q15_t *outp = outputQ15.ptr();
       
      
       arm_copy_q15(inp, outp,this->nbSamples);
         
          
       ASSERT_EQ(inputQ15,outputQ15);
       ASSERT_EMPTY_TAIL(outputQ15);

    } 

    void SupportTestsQ15::test_fill_q15()
    {
       q15_t *outp = outputQ15.ptr();
       q15_t val = 0x4000;
       int i;
      

       arm_fill_q15(val, outp,this->nbSamples);
         
          
       for(i=0 ; i < this->nbSamples; i++)
       {
          ASSERT_EQ(val,outp[i]);
       }
       ASSERT_EMPTY_TAIL(outputQ15);

    } 

    void SupportTestsQ15::test_q15_float()
    {
       const q15_t *inp = inputQ15.ptr();
       float32_t *refp = refF32.ptr();
       float32_t *outp = outputF32.ptr();
       
      
       arm_q15_to_float(inp, outp,this->nbSamples);
         
          
       ASSERT_REL_ERROR(refF32,outputF32,REL_ERROR);
       ASSERT_EMPTY_TAIL(outputF32);

    } 

    void SupportTestsQ15::test_q15_q31()
    {
       const q15_t *inp = inputQ15.ptr();
       q31_t *refp = refQ31.ptr();
       q31_t *outp = outputQ31.ptr();
       
      
       arm_q15_to_q31(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ31,outputQ31,ABS_Q31_ERROR);
       ASSERT_EMPTY_TAIL(outputQ31);

    } 

    void SupportTestsQ15::test_q15_q7()
    {
       const q15_t *inp = inputQ15.ptr();
       q7_t *refp = refQ7.ptr();
       q7_t *outp = outputQ7.ptr();
       
      
       arm_q15_to_q7(inp, outp,this->nbSamples);
         
          
       ASSERT_NEAR_EQ(refQ7,outputQ7,ABS_Q7_ERROR);
       ASSERT_EMPTY_TAIL(outputQ7);

    } 

    __ALIGNED(2) static const q15_t testReadQ15[2]={-2,1};
    __ALIGNED(2) static q15_t testWriteQ15[2]={0,0};

    void SupportTestsQ15::test_read_q15x2()
    {
        q31_t result=0;

        result = read_q15x2((q15_t*)testReadQ15);

        printf("%08X\n",result);

        ASSERT_TRUE(result == 0x0001FFFE);

    }

    void SupportTestsQ15::test_read_q15x2_ia()
    {
        q31_t result=0;
        q15_t *p = (q15_t*)testReadQ15;

        result = read_q15x2_ia(&p);

        ASSERT_TRUE(result == 0x0001FFFE);
        ASSERT_TRUE(p == testReadQ15 + 2);
    }

    void SupportTestsQ15::test_read_q15x2_da()
    {
        q31_t result=0;
        q15_t *p = (q15_t*)testReadQ15;

        result = read_q15x2_da(&p);

        ASSERT_TRUE(result == 0x0001FFFE);
        ASSERT_TRUE(p == testReadQ15 - 2);
    }

    void SupportTestsQ15::test_write_q15x2_ia()
    {
         q31_t val = 0x0001FFFE;
         q15_t *p = testWriteQ15;

         testWriteQ15[0] = 0;
         testWriteQ15[1] = 0;

         write_q15x2_ia(&p,val);

         ASSERT_TRUE(testWriteQ15[0] == -2);
         ASSERT_TRUE(testWriteQ15[1] == 1);
         ASSERT_TRUE(p == testWriteQ15 + 2);

    }

    void SupportTestsQ15::test_write_q15x2()
    {
         q31_t val = 0x0001FFFE;

         testWriteQ15[0] = 0;
         testWriteQ15[1] = 0;

         write_q15x2(testWriteQ15,val);

         ASSERT_TRUE(testWriteQ15[0] == -2);
         ASSERT_TRUE(testWriteQ15[1] == 1);
    }

  
    void SupportTestsQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        switch(id)
        {
 
            case TEST_COPY_Q15_1:
              this->nbSamples = 7;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);

              outputQ15.create(inputQ15.nbSamples(),SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_COPY_Q15_2:
              this->nbSamples = 16;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);

              outputQ15.create(inputQ15.nbSamples(),SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_COPY_Q15_3:
              this->nbSamples = 23;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);

              outputQ15.create(inputQ15.nbSamples(),SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_FILL_Q15_4:
              this->nbSamples = 7;

              outputQ15.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_FILL_Q15_5:
              this->nbSamples = 16;

              outputQ15.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_FILL_Q15_6:
              this->nbSamples = 23;

              outputQ15.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_FLOAT_7:
              this->nbSamples = 7;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ15::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_FLOAT_8:
              this->nbSamples = 16;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ15::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_FLOAT_9:
              this->nbSamples = 23;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refF32.reload(SupportTestsQ15::SAMPLES_F32_ID,mgr,this->nbSamples);
              outputF32.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_Q31_10:
              this->nbSamples = 7;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsQ15::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_Q31_11:
              this->nbSamples = 16;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsQ15::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_Q31_12:
              this->nbSamples = 23;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refQ31.reload(SupportTestsQ15::SAMPLES_Q31_ID,mgr,this->nbSamples);
              outputQ31.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_Q7_13:
              this->nbSamples = 7;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsQ15::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_Q7_14:
              this->nbSamples = 16;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsQ15::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

            case TEST_Q15_Q7_15:
              this->nbSamples = 23;
              inputQ15.reload(SupportTestsQ15::SAMPLES_Q15_ID,mgr,this->nbSamples);
              refQ7.reload(SupportTestsQ15::SAMPLES_Q7_ID,mgr,this->nbSamples);
              outputQ7.create(this->nbSamples,SupportTestsQ15::OUT_ID,mgr);

            break;

        }

       

    }

    void SupportTestsQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      switch(id)
      {
 
            case TEST_COPY_Q15_1:
            case TEST_COPY_Q15_2:
            case TEST_COPY_Q15_3:
            case TEST_FILL_Q15_4:
            case TEST_FILL_Q15_5:
            case TEST_FILL_Q15_6:
               outputQ15.dump(mgr);
            break;

            case TEST_Q15_FLOAT_7:
            case TEST_Q15_FLOAT_8:
            case TEST_Q15_FLOAT_9:
               outputF32.dump(mgr);
            break;

            case TEST_Q15_Q31_10:
            case TEST_Q15_Q31_11:
            case TEST_Q15_Q31_12:
               outputQ31.dump(mgr);
            break;

            case TEST_Q15_Q7_13:
            case TEST_Q15_Q7_14:
            case TEST_Q15_Q7_15:
               outputQ7.dump(mgr);
            break;
      }
    }
