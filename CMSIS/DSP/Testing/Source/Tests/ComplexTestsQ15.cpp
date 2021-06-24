#include "ComplexTestsQ15.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 25
#define SNR_HIGH_THRESHOLD 60

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)50)
#define ABS_ERROR_Q31 ((q31_t)(1<<15))

    void ComplexTestsQ15::test_cmplx_conj_q15()
    {
        const q15_t *inp1=input1.ptr();
        q15_t *outp=output.ptr();

        arm_cmplx_conj_q15(inp1,outp,input1.nbSamples() >> 1  );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 


    void ComplexTestsQ15::test_cmplx_dot_prod_q15()
    {
        q31_t re,im;

        const q15_t *inp1=input1.ptr();
        const q15_t *inp2=input2.ptr();
        q31_t *outp=dotOutput.ptr();

        arm_cmplx_dot_prod_q15(inp1,inp2,input1.nbSamples() >> 1  ,&re,&im);

        outp[0] = re;
        outp[1] = im;

        ASSERT_SNR(dotOutput,dotRef,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(dotOutput,dotRef,ABS_ERROR_Q31);

        ASSERT_EMPTY_TAIL(dotOutput);

       
    } 

    void ComplexTestsQ15::test_cmplx_mag_q15()
    {
        const q15_t *inp1=input1.ptr();
        q15_t *outp=output.ptr();

        arm_cmplx_mag_q15(inp1,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_HIGH_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

    void ComplexTestsQ15::test_cmplx_mag_squared_q15()
    {
        const q15_t *inp1=input1.ptr();
        q15_t *outp=output.ptr();

        arm_cmplx_mag_squared_q15(inp1,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

    void ComplexTestsQ15::test_cmplx_mult_cmplx_q15()
    {
        const q15_t *inp1=input1.ptr();
        const q15_t *inp2=input2.ptr();
        q15_t *outp=output.ptr();

        arm_cmplx_mult_cmplx_q15(inp1,inp2,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

    void ComplexTestsQ15::test_cmplx_mult_real_q15()
    {
        const q15_t *inp1=input1.ptr();
        const q15_t *inp2=input2.ptr();
        q15_t *outp=output.ptr();

        arm_cmplx_mult_real_q15(inp1,inp2,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 
 
    void ComplexTestsQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case ComplexTestsQ15::TEST_CMPLX_CONJ_Q15_1:
          nb = 7;
          ref.reload(ComplexTestsQ15::REF_CONJ_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_CONJ_Q15_2:
          nb = 16;
          ref.reload(ComplexTestsQ15::REF_CONJ_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_CONJ_Q15_3:
          nb = 23;
          ref.reload(ComplexTestsQ15::REF_CONJ_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_DOT_PROD_Q15_4:
          nb = 7;
          dotRef.reload(ComplexTestsQ15::REF_DOT_PROD_3_Q15_ID,mgr);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT2_Q15_ID,mgr,nb << 1);

          dotOutput.create(dotRef.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;

        case ComplexTestsQ15::TEST_CMPLX_DOT_PROD_Q15_5:
          nb = 16;
          dotRef.reload(ComplexTestsQ15::REF_DOT_PROD_4N_Q15_ID,mgr);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT2_Q15_ID,mgr,nb << 1);

          dotOutput.create(dotRef.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;

        case ComplexTestsQ15::TEST_CMPLX_DOT_PROD_Q15_6:
          nb = 23;
          dotRef.reload(ComplexTestsQ15::REF_DOT_PROD_4N1_Q15_ID,mgr);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT2_Q15_ID,mgr,nb << 1);

          dotOutput.create(dotRef.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MAG_Q15_7:
          nb = 7;
          ref.reload(ComplexTestsQ15::REF_MAG_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MAG_Q15_8:
          nb = 16;
          ref.reload(ComplexTestsQ15::REF_MAG_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MAG_Q15_9:
          nb = 23;
          ref.reload(ComplexTestsQ15::REF_MAG_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MAG_SQUARED_Q15_10:
          nb = 7;
          ref.reload(ComplexTestsQ15::REF_MAG_SQUARED_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MAG_SQUARED_Q15_11:
          nb = 16;
          ref.reload(ComplexTestsQ15::REF_MAG_SQUARED_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MAG_SQUARED_Q15_12:
          nb = 23;
          ref.reload(ComplexTestsQ15::REF_MAG_SQUARED_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MULT_CMPLX_Q15_13:
          nb = 7;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_CMPLX_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT2_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MULT_CMPLX_Q15_14:
          nb = 16;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_CMPLX_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT2_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MULT_CMPLX_Q15_15:
          nb = 23;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_CMPLX_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT2_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MULT_REAL_Q15_16:
          nb = 7;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_REAL_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT3_Q15_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MULT_REAL_Q15_17:
          nb = 16;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_REAL_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT3_Q15_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
        case ComplexTestsQ15::TEST_CMPLX_MULT_REAL_Q15_18:
          nb = 23;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_REAL_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT3_Q15_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;

        case ComplexTestsQ15::TEST_CMPLX_CONJ_Q15_19:
          nb = 256;
          ref.reload(ComplexTestsQ15::REF_CONJ_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
        break;

        case ComplexTestsQ15::TEST_CMPLX_MAG_Q15_20:
          nb = 256;
          ref.reload(ComplexTestsQ15::REF_MAG_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
        break;
        
        case ComplexTestsQ15::TEST_CMPLX_MAG_SQUARED_Q15_21:
          nb = 256;
          ref.reload(ComplexTestsQ15::REF_MAG_SQUARED_Q15_ID,mgr,nb);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
        break;
        
        case ComplexTestsQ15::TEST_CMPLX_MULT_CMPLX_Q15_22:
          nb = 256;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_CMPLX_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT2_Q15_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
        break;
        
        case ComplexTestsQ15::TEST_CMPLX_MULT_REAL_Q15_23:
          nb = 256;
          ref.reload(ComplexTestsQ15::REF_CMPLX_MULT_REAL_Q15_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ15::INPUT1_Q15_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ15::INPUT3_Q15_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
        break;
       }
      

       
    }

    void ComplexTestsQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       switch(id)
       {
         case ComplexTestsQ15::TEST_CMPLX_DOT_PROD_Q15_4:
         case ComplexTestsQ15::TEST_CMPLX_DOT_PROD_Q15_5:
         case ComplexTestsQ15::TEST_CMPLX_DOT_PROD_Q15_6:
            dotOutput.dump(mgr);
         break;

         default:
            output.dump(mgr);
       }
    }
