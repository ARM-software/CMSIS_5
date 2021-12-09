#include "ComplexTestsF64.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 310

#define REL_ERROR (3.0e-15)

/*
    void ComplexTestsF64::test_cmplx_conj_f64()
    {
        const float64_t *inp1=input1.ptr();
        float64_t *outp=output.ptr();


        arm_cmplx_conj_f64(inp1,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);


    } 


    void ComplexTestsF64::test_cmplx_dot_prod_f64()
    {
        float64_t re,im;

        const float64_t *inp1=input1.ptr();
        const float64_t *inp2=input2.ptr();
        float64_t *outp=output.ptr();

        arm_cmplx_dot_prod_f64(inp1,inp2,input1.nbSamples() >> 1,&re,&im);

        outp[0] = re;
        outp[1] = im;

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

        ASSERT_EMPTY_TAIL(output);
    } 
*/
    void ComplexTestsF64::test_cmplx_mag_f64()
    {
        const float64_t *inp1=input1.ptr();
        float64_t *outp=output.ptr();

        arm_cmplx_mag_f64(inp1,outp,input1.nbSamples() >> 1 );
        
        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF64::test_cmplx_mag_squared_f64()
    {
        const float64_t *inp1=input1.ptr();
        float64_t *outp=output.ptr();

        arm_cmplx_mag_squared_f64(inp1,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF64::test_cmplx_mult_cmplx_f64()
    {
        const float64_t *inp1=input1.ptr();
        const float64_t *inp2=input2.ptr();
        float64_t *outp=output.ptr();

        arm_cmplx_mult_cmplx_f64(inp1,inp2,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

/*
    void ComplexTestsF64::test_cmplx_mult_real_f64()
    {
        const float64_t *inp1=input1.ptr();
        const float64_t *inp2=input2.ptr();
        float64_t *outp=output.ptr();

        arm_cmplx_mult_real_f64(inp1,inp2,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 
 */
    void ComplexTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       (void)params;

       
       switch(id)
       {
        case ComplexTestsF64::TEST_CMPLX_CONJ_F64_1:
          nb = 2;
          ref.reload(ComplexTestsF64::REF_CONJ_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_CONJ_F64_2:
          nb = 4;
          ref.reload(ComplexTestsF64::REF_CONJ_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_CONJ_F64_3:
          nb = 5;
          ref.reload(ComplexTestsF64::REF_CONJ_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_DOT_PROD_F64_4:
          nb = 2;
          ref.reload(ComplexTestsF64::REF_DOT_PROD_3_F64_ID,mgr);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;

        case ComplexTestsF64::TEST_CMPLX_DOT_PROD_F64_5:
          nb = 4;
          ref.reload(ComplexTestsF64::REF_DOT_PROD_4N_F64_ID,mgr);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;

        case ComplexTestsF64::TEST_CMPLX_DOT_PROD_F64_6:
          nb = 5;
          ref.reload(ComplexTestsF64::REF_DOT_PROD_4N1_F64_ID,mgr);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MAG_F64_7:
          nb = 2;
          ref.reload(ComplexTestsF64::REF_MAG_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MAG_F64_8:
          nb = 4;
          ref.reload(ComplexTestsF64::REF_MAG_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MAG_F64_9:
          nb = 5;
          ref.reload(ComplexTestsF64::REF_MAG_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MAG_SQUARED_F64_10:
          nb = 2;
          ref.reload(ComplexTestsF64::REF_MAG_SQUARED_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MAG_SQUARED_F64_11:
          nb = 4;
          ref.reload(ComplexTestsF64::REF_MAG_SQUARED_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MAG_SQUARED_F64_12:
          nb = 5;
          ref.reload(ComplexTestsF64::REF_MAG_SQUARED_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MULT_CMPLX_F64_13:
          nb = 2;
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_CMPLX_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MULT_CMPLX_F64_14:
          nb = 4;
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_CMPLX_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MULT_CMPLX_F64_15:
          nb = 5;
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_CMPLX_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MULT_REAL_F64_16:
          nb = 2;
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_REAL_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT3_F64_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MULT_REAL_F64_17:
          nb = 4;
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_REAL_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT3_F64_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;
        case ComplexTestsF64::TEST_CMPLX_MULT_REAL_F64_18:
          nb = 5;
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_REAL_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT3_F64_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
          break;

        case ComplexTestsF64::TEST_CMPLX_CONJ_F64_19:
          ref.reload(ComplexTestsF64::REF_CONJ_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
        break;

        case ComplexTestsF64::TEST_CMPLX_DOT_PROD_F64_20:
          ref.reload(ComplexTestsF64::REF_DOT_PROD_LONG_F64_ID,mgr);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
        break;
        
        case ComplexTestsF64::TEST_CMPLX_MAG_F64_21:
          ref.reload(ComplexTestsF64::REF_MAG_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
        break;
        
        case ComplexTestsF64::TEST_CMPLX_MAG_SQUARED_F64_22:
          ref.reload(ComplexTestsF64::REF_MAG_SQUARED_F64_ID,mgr,nb);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
        break;
        
        case ComplexTestsF64::TEST_CMPLX_MULT_CMPLX_F64_23:
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_CMPLX_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT2_F64_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
        break;
        
        case ComplexTestsF64::TEST_CMPLX_MULT_REAL_F64_24:
          ref.reload(ComplexTestsF64::REF_CMPLX_MULT_REAL_F64_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF64::INPUT1_F64_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF64::INPUT3_F64_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF64::OUT_SAMPLES_F64_ID,mgr);
        break;
        
       }
      
    }

    void ComplexTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
