#include "BasicTestsQ31.h"
#include "Error.h"

#define SNR_THRESHOLD 100

#define ONEHALF 0x40000000

#define GET_Q31_PTR() \
const q31_t *inp1=input1.ptr(); \
const q31_t *inp2=input2.ptr(); \
q31_t *refp=ref.ptr(); \
q31_t *outp=output.ptr();

    void BasicTestsQ31::test_add_q31()
    {
        GET_Q31_PTR();

        arm_add_q31(inp1,inp2,outp,input1.nbSamples());
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

    } 

    void BasicTestsQ31::test_sub_q31()
    {
        GET_Q31_PTR();

        arm_sub_q31(inp1,inp2,outp,input1.nbSamples());
        
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ31::test_mult_q31()
    {
        GET_Q31_PTR();

        arm_mult_q31(inp1,inp2,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ31::test_negate_q31()
    {
        const q31_t *inp1=input1.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_negate_q31(inp1,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ31::test_offset_q31()
    {
        const q31_t *inp1=input1.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_offset_q31(inp1,this->scalar,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ31::test_scale_q31()
    {
        const q31_t *inp1=input1.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_scale_q31(inp1,this->scalar,0,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ31::test_dot_prod_q31()
    {
        q63_t r;

        const q31_t *inp1=input1.ptr();
        const q31_t *inp2=input2.ptr();
        q63_t *refp=dotRef.ptr(); 
        q63_t *outp=dotOutput.ptr();

        arm_dot_prod_q31(inp1,inp2,input1.nbSamples(),&r);

        outp[0] = r;

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       
    } 

    void BasicTestsQ31::test_abs_q31()
    {
        GET_Q31_PTR();

        arm_abs_q31(inp1,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 

    void BasicTestsQ31::test_shift_q31()
    {
        const q31_t *inp1=input1.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_shift_q31(inp1,1,outp,input1.nbSamples());

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
       
    } 


    void BasicTestsQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       this->scalar = ONEHALF;

       
       switch(id)
       {
        case BasicTestsQ31::TEST_ADD_Q31_1:
          nb = 3;
          ref.reload(BasicTestsQ31::REF_ADD_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_ADD_Q31_2:
          nb = 8;
          ref.reload(BasicTestsQ31::REF_ADD_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_ADD_Q31_3:
          nb = 9;
          ref.reload(BasicTestsQ31::REF_ADD_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;


        case BasicTestsQ31::TEST_SUB_Q31_4:
          nb = 3;
          ref.reload(BasicTestsQ31::REF_SUB_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_SUB_Q31_5:
          nb = 8;
          ref.reload(BasicTestsQ31::REF_SUB_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_SUB_Q31_6:
          nb = 9;
          ref.reload(BasicTestsQ31::REF_SUB_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_MULT_Q31_7:
          nb = 3;
          ref.reload(BasicTestsQ31::REF_MULT_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_MULT_Q31_8:
          nb = 8;
          ref.reload(BasicTestsQ31::REF_MULT_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_MULT_Q31_9:
          nb = 9;
          ref.reload(BasicTestsQ31::REF_MULT_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_NEGATE_Q31_10:
          nb = 3;
          ref.reload(BasicTestsQ31::REF_NEGATE_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_NEGATE_Q31_11:
          nb = 8;
          ref.reload(BasicTestsQ31::REF_NEGATE_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_NEGATE_Q31_12:
          nb = 9;
          ref.reload(BasicTestsQ31::REF_NEGATE_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_OFFSET_Q31_13:
          nb = 3;
          ref.reload(BasicTestsQ31::REF_OFFSET_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_OFFSET_Q31_14:
          nb = 8;
          ref.reload(BasicTestsQ31::REF_OFFSET_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_OFFSET_Q31_15:
          nb = 9;
          ref.reload(BasicTestsQ31::REF_OFFSET_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_SCALE_Q31_16:
          nb = 3;
          ref.reload(BasicTestsQ31::REF_SCALE_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_SCALE_Q31_17:
          nb = 8;
          ref.reload(BasicTestsQ31::REF_SCALE_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_SCALE_Q31_18:
          nb = 9;
          ref.reload(BasicTestsQ31::REF_SCALE_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_DOT_PROD_Q31_19:
          nb = 3;
          dotRef.reload(BasicTestsQ31::REF_DOT_3_Q31_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_DOT_PROD_Q31_20:
          nb = 8;
          dotRef.reload(BasicTestsQ31::REF_DOT_4N_Q31_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_DOT_PROD_Q31_21:
          nb = 9;
          dotRef.reload(BasicTestsQ31::REF_DOT_4N1_Q31_ID,mgr);
          dotOutput.create(dotRef.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_ABS_Q31_22:
          nb = 3;
          ref.reload(BasicTestsQ31::REF_ABS_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_ABS_Q31_23:
          nb = 8;
          ref.reload(BasicTestsQ31::REF_ABS_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;
        case BasicTestsQ31::TEST_ABS_Q31_24:
          nb = 9;
          ref.reload(BasicTestsQ31::REF_ABS_Q31_ID,mgr,nb);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUT1_Q31_ID,mgr,nb);
          input2.reload(BasicTestsQ31::INPUT2_Q31_ID,mgr,nb);
          break;

        case BasicTestsQ31::TEST_ADD_Q31_25:
          input1.reload(BasicTestsQ31::MAXPOS_Q31_ID,mgr);
          input2.reload(BasicTestsQ31::MAXPOS_Q31_ID,mgr);
          ref.reload(BasicTestsQ31::REF_POSSAT_12_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
        break;

        case BasicTestsQ31::TEST_ADD_Q31_26:
          input1.reload(BasicTestsQ31::MAXNEG_Q31_ID,mgr);
          input2.reload(BasicTestsQ31::MAXNEG_Q31_ID,mgr);
          ref.reload(BasicTestsQ31::REF_NEGSAT_13_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
        break;

        case BasicTestsQ31::TEST_SUB_Q31_27:
          ref.reload(BasicTestsQ31::REF_POSSAT_14_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXPOS_Q31_ID,mgr);
          input2.reload(BasicTestsQ31::MAXNEG_Q31_ID,mgr);
        break;

        case BasicTestsQ31::TEST_SUB_Q31_28:
          ref.reload(BasicTestsQ31::REF_NEGSAT_15_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXNEG_Q31_ID,mgr);
          input2.reload(BasicTestsQ31::MAXPOS_Q31_ID,mgr);
        break;

        case BasicTestsQ31::TEST_MULT_Q31_29:
          ref.reload(BasicTestsQ31::REF_POSSAT_16_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXNEG2_Q31_ID,mgr);
          input2.reload(BasicTestsQ31::MAXNEG2_Q31_ID,mgr);
        break;

        case BasicTestsQ31::TEST_NEGATE_Q31_30:
          ref.reload(BasicTestsQ31::REF_POSSAT_17_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXNEG2_Q31_ID,mgr);
          break;

        case BasicTestsQ31::TEST_OFFSET_Q31_31:
          ref.reload(BasicTestsQ31::REF_POSSAT_18_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXPOS_Q31_ID,mgr);
          /* 0.9 */
          this->scalar = 0x73333333;
          break;

        case BasicTestsQ31::TEST_OFFSET_Q31_32:
          ref.reload(BasicTestsQ31::REF_NEGSAT_19_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXNEG_Q31_ID,mgr);
          /* -0.9 */
          this->scalar = 0x8ccccccd;
          break;

        case BasicTestsQ31::TEST_SCALE_Q31_33:
          ref.reload(BasicTestsQ31::REF_POSSAT_20_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXNEG2_Q31_ID,mgr);
          /* Minus max*/
          this->scalar = 0x80000000;
          break;

        case BasicTestsQ31::TEST_SHIFT_Q31_34:
          ref.reload(BasicTestsQ31::REF_SHIFT_21_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::INPUTRAND_Q31_ID,mgr);
        break;

        case BasicTestsQ31::TEST_SHIFT_Q31_35:
          ref.reload(BasicTestsQ31::REF_SHIFT_POSSAT_22_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXPOS_Q31_ID,mgr);
        break;

        case BasicTestsQ31::TEST_SHIFT_Q31_36:
          ref.reload(BasicTestsQ31::REF_SHIFT_NEGSAT_23_Q31_ID,mgr);
          output.create(ref.nbSamples(),BasicTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
          input1.reload(BasicTestsQ31::MAXNEG_Q31_ID,mgr);
        break;

       }
      

       

    }

    void BasicTestsQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       switch(id)
       {
         case BasicTestsQ31::TEST_DOT_PROD_Q31_19:
         case BasicTestsQ31::TEST_DOT_PROD_Q31_20:
         case BasicTestsQ31::TEST_DOT_PROD_Q31_21:
            dotOutput.dump(mgr);
         break;

         default:
            output.dump(mgr);
       }

        
    }
