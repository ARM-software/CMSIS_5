#include "ComplexTestsQ31.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 100

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)100)
#define ABS_ERROR_Q63 ((q63_t)(1<<18))


    void ComplexTestsQ31::test_cmplx_conj_q31()
    {
        const q31_t *inp1=input1.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_cmplx_conj_q31(inp1,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 


    void ComplexTestsQ31::test_cmplx_dot_prod_q31()
    {
        q63_t re,im;

        const q31_t *inp1=input1.ptr();
        const q31_t *inp2=input2.ptr();
        q63_t *outp=dotOutput.ptr();

        arm_cmplx_dot_prod_q31(inp1,inp2,input1.nbSamples() >> 1  ,&re,&im);

        outp[0] = re;
        outp[1] = im;

        ASSERT_SNR(dotOutput,dotRef,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(dotOutput,dotRef,ABS_ERROR_Q63);

         ASSERT_EMPTY_TAIL(dotOutput);

       
    } 

    void ComplexTestsQ31::test_cmplx_mag_q31()
    {
        const q31_t *inp1=input1.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_cmplx_mag_q31(inp1,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

    void ComplexTestsQ31::test_cmplx_mag_squared_q31()
    {
        const q31_t *inp1=input1.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_cmplx_mag_squared_q31(inp1,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

    void ComplexTestsQ31::test_cmplx_mult_cmplx_q31()
    {
        const q31_t *inp1=input1.ptr();
        const q31_t *inp2=input2.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_cmplx_mult_cmplx_q31(inp1,inp2,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

    void ComplexTestsQ31::test_cmplx_mult_real_q31()
    {
        const q31_t *inp1=input1.ptr();
        const q31_t *inp2=input2.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_cmplx_mult_real_q31(inp1,inp2,outp,input1.nbSamples()  >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 
 
    void ComplexTestsQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case ComplexTestsQ31::TEST_CMPLX_CONJ_Q31_1:
          nb = 3;
          ref.reload(ComplexTestsQ31::REF_CONJ_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_CONJ_Q31_2:
          nb = 8;
          ref.reload(ComplexTestsQ31::REF_CONJ_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_CONJ_Q31_3:
          nb = 11;
          ref.reload(ComplexTestsQ31::REF_CONJ_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_DOT_PROD_Q31_4:
          nb = 3;
          dotRef.reload(ComplexTestsQ31::REF_DOT_PROD_3_Q31_ID,mgr);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT2_Q31_ID,mgr,nb << 1);

          dotOutput.create(dotRef.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;

        case ComplexTestsQ31::TEST_CMPLX_DOT_PROD_Q31_5:
          nb = 8;
          dotRef.reload(ComplexTestsQ31::REF_DOT_PROD_4N_Q31_ID,mgr);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT2_Q31_ID,mgr,nb << 1);

          dotOutput.create(dotRef.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;

        case ComplexTestsQ31::TEST_CMPLX_DOT_PROD_Q31_6:
          nb = 11;
          dotRef.reload(ComplexTestsQ31::REF_DOT_PROD_4N1_Q31_ID,mgr);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT2_Q31_ID,mgr,nb << 1);

          dotOutput.create(dotRef.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MAG_Q31_7:
          nb = 3;
          ref.reload(ComplexTestsQ31::REF_MAG_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MAG_Q31_8:
          nb = 8;
          ref.reload(ComplexTestsQ31::REF_MAG_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MAG_Q31_9:
          nb = 11;
          ref.reload(ComplexTestsQ31::REF_MAG_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MAG_SQUARED_Q31_10:
          nb = 3;
          ref.reload(ComplexTestsQ31::REF_MAG_SQUARED_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MAG_SQUARED_Q31_11:
          nb = 8;
          ref.reload(ComplexTestsQ31::REF_MAG_SQUARED_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MAG_SQUARED_Q31_12:
          nb = 11;
          ref.reload(ComplexTestsQ31::REF_MAG_SQUARED_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MULT_CMPLX_Q31_13:
          nb = 3;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_CMPLX_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT2_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MULT_CMPLX_Q31_14:
          nb = 8;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_CMPLX_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT2_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MULT_CMPLX_Q31_15:
          nb = 11;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_CMPLX_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT2_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MULT_REAL_Q31_16:
          nb = 3;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_REAL_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT3_Q31_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MULT_REAL_Q31_17:
          nb = 8;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_REAL_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT3_Q31_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
        case ComplexTestsQ31::TEST_CMPLX_MULT_REAL_Q31_18:
          nb = 11;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_REAL_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT3_Q31_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;

        case ComplexTestsQ31::TEST_CMPLX_CONJ_Q31_19:
          nb = 256;
          ref.reload(ComplexTestsQ31::REF_CONJ_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
        break;

        case ComplexTestsQ31::TEST_CMPLX_MAG_Q31_20:
          nb = 256;
          ref.reload(ComplexTestsQ31::REF_MAG_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
        break;
        
        case ComplexTestsQ31::TEST_CMPLX_MAG_SQUARED_Q31_21:
          nb = 256;
          ref.reload(ComplexTestsQ31::REF_MAG_SQUARED_Q31_ID,mgr,nb);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
        break;
        
        case ComplexTestsQ31::TEST_CMPLX_MULT_CMPLX_Q31_22:
          nb = 256;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_CMPLX_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT2_Q31_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
        break;
        
        case ComplexTestsQ31::TEST_CMPLX_MULT_REAL_Q31_23:
          nb = 256;
          ref.reload(ComplexTestsQ31::REF_CMPLX_MULT_REAL_Q31_ID,mgr,nb << 1);
          input1.reload(ComplexTestsQ31::INPUT1_Q31_ID,mgr,nb << 1);
          input2.reload(ComplexTestsQ31::INPUT3_Q31_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
        break;
        
       }
      

       
    }

    void ComplexTestsQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       switch(id)
       {
         case ComplexTestsQ31::TEST_CMPLX_DOT_PROD_Q31_4:
         case ComplexTestsQ31::TEST_CMPLX_DOT_PROD_Q31_5:
         case ComplexTestsQ31::TEST_CMPLX_DOT_PROD_Q31_6:
            dotOutput.dump(mgr);
         break;

         default:
            output.dump(mgr);
       }
    }
