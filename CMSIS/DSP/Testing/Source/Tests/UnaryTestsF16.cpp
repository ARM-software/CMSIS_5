#include "UnaryTestsF16.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 59

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.1e-3)
#define ABS_ERROR (1.1e-3)

/*

Comparisons for inverse

*/

/* Not very accurate for big matrix.
But big matrix needed for checking the vectorized code */

#define SNR_THRESHOLD_INV 45
#define REL_ERROR_INV (3.0e-2)
#define ABS_ERROR_INV (3.0e-2)

#define REL_ERROR_SOLVE (6.0e-3)
#define ABS_ERROR_SOLVE (6.0e-2)

/*

Comparison for Cholesky

*/
#define SNR_THRESHOLD_CHOL 45
#define REL_ERROR_CHOL (3.0e-3)
#define ABS_ERROR_CHOL (3.0e-2)

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 40

#define LOADDATA2()                          \
      const float16_t *inp1=input1.ptr();    \
      const float16_t *inp2=input2.ptr();    \
                                             \
      float16_t *ap=a.ptr();                 \
      float16_t *bp=b.ptr();                 \
                                             \
      float16_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define LOADDATA1()                          \
      const float16_t *inp1=input1.ptr();    \
                                             \
      float16_t *ap=a.ptr();                 \
                                             \
      float16_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float16_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=rows;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,sizeof(float16_t)*rows*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

#define PREPAREDATA1(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float16_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      if (TRANSPOSED)                                                    \
      {                                                                  \
         out.numRows=columns;                                            \
         out.numCols=rows;                                               \
      }                                                                  \
      else                                                               \
      {                                                                  \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      }                                                                  \
      out.pData = outp;

#define PREPAREDATA1C(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,2*sizeof(float16_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      if (TRANSPOSED)                                                    \
      {                                                                  \
         out.numRows=columns;                                            \
         out.numCols=rows;                                               \
      }                                                                  \
      else                                                               \
      {                                                                  \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      }                                                                  \
      out.pData = outp;

#define LOADVECDATA2()                          \
      const float16_t *inp1=input1.ptr();    \
      const float16_t *inp2=input2.ptr();    \
                                             \
      float16_t *ap=a.ptr();                 \
      float16_t *bp=b.ptr();                 \
                                             \
      float16_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() / 2;\
      int rows,internal;                      \
      int i;

#define PREPAREVECDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=internal;                                               \
      memcpy((void*)ap,(const void*)inp1,2*sizeof(float16_t)*rows*internal);\
      in1.pData = ap;                                                    \
                                                                         \
      memcpy((void*)bp,(const void*)inp2,2*sizeof(float16_t)*internal);
                            


void UnaryTestsF16::test_mat_vec_mult_f16()
    {     
      LOADVECDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;

          PREPAREVECDATA2();

          arm_mat_vec_mult_f16(&this->in1, bp, outp);

          outp += rows ;

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void UnaryTestsF16::test_mat_add_f16()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_add_f16(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF16::test_mat_sub_f16()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_sub_f16(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF16::test_mat_scale_f16()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(false);

          status=arm_mat_scale_f16(&this->in1,0.5f,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF16::test_mat_trans_f16()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(true);

          status=arm_mat_trans_f16(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF16::test_mat_cmplx_trans_f16()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1C(true);

          status=arm_mat_cmplx_trans_f16(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += 2*(rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    }

void UnaryTestsF16::test_mat_inverse_f16()
    {     
      const float16_t *inp1=input1.ptr();    
                                             
      float16_t *ap=a.ptr();                 
                                             
      float16_t *outp=output.ptr();          
      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();
      int rows,columns;                      
      int i;
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATA1(false);

          status=arm_mat_inverse_f16(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD_INV);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR_INV,REL_ERROR_INV);

    }

    void UnaryTestsF16::test_mat_cholesky_dpo_f16()
    {
      float16_t *ap=a.ptr();                 
      const float16_t *inp1=input1.ptr();    
                                             
                                             
      float16_t *outp=output.ptr();     
      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();

      int rows,columns;                      
      int i;
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATA1(false);

          status=arm_mat_cholesky_f16(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD_CHOL);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR_CHOL,REL_ERROR_CHOL);
    }

    void UnaryTestsF16::test_solve_upper_triangular_f16()
    {
      float16_t *ap=a.ptr();                 
      const float16_t *inp1=input1.ptr();    

      float16_t *bp=b.ptr();                 
      const float16_t *inp2=input2.ptr();    
                                             
                                             
      float16_t *outp=output.ptr();     
      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();

      int rows,columns;                      
      int i;
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATA2();

          status=arm_mat_solve_upper_triangular_f16(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);
          inp2 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR_SOLVE,REL_ERROR_SOLVE);
    }

    void UnaryTestsF16::test_solve_lower_triangular_f16()
    {
      float16_t *ap=a.ptr();                 
      const float16_t *inp1=input1.ptr();    

      float16_t *bp=b.ptr();                 
      const float16_t *inp2=input2.ptr();    
                                             
                                             
      float16_t *outp=output.ptr();     
      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();

      int rows,columns;                      
      int i;
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATA2();

          status=arm_mat_solve_lower_triangular_f16(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);
          inp2 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR_SOLVE,REL_ERROR_SOLVE);
    }

    void UnaryTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


      (void)params;
      switch(id)
      {
         case TEST_MAT_ADD_F16_1:
            input1.reload(UnaryTestsF16::INPUTS1_F16_ID,mgr);
            input2.reload(UnaryTestsF16::INPUTS2_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFADD1_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPB_F16_ID,mgr);
         break;

         case TEST_MAT_SUB_F16_2:
            input1.reload(UnaryTestsF16::INPUTS1_F16_ID,mgr);
            input2.reload(UnaryTestsF16::INPUTS2_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFSUB1_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPB_F16_ID,mgr);
         break;

         case TEST_MAT_SCALE_F16_3:
            input1.reload(UnaryTestsF16::INPUTS1_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFSCALE1_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
         break;

         case TEST_MAT_TRANS_F16_4:
            input1.reload(UnaryTestsF16::INPUTS1_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFTRANS1_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
         break;

         case TEST_MAT_INVERSE_F16_5:
            input1.reload(UnaryTestsF16::INPUTSINV_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSINVERT1_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFINV1_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
         break;

         case TEST_MAT_VEC_MULT_F16_6:
            input1.reload(UnaryTestsF16::INPUTS1_F16_ID,mgr);
            input2.reload(UnaryTestsF16::INPUTVEC1_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFVECMUL1_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
            b.create(MAXMATRIXDIM,UnaryTestsF16::TMPB_F16_ID,mgr);
         break;

          case TEST_MAT_CMPLX_TRANS_F16_7:
            input1.reload(UnaryTestsF16::INPUTSC1_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFTRANSC1_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
         break;

         case TEST_MAT_CHOLESKY_DPO_F16_8:
            input1.reload(UnaryTestsF16::INPUTSCHOLESKY1_DPO_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSCHOLESKY1_DPO_S16_ID,mgr);

            ref.reload(UnaryTestsF16::REFCHOLESKY1_DPO_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
         break;

         case TEST_SOLVE_UPPER_TRIANGULAR_F16_9:
            input1.reload(UnaryTestsF16::INPUT_UT_DPO_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSCHOLESKY1_DPO_S16_ID,mgr);
            input2.reload(UnaryTestsF16::INPUT_RNDA_DPO_F16_ID,mgr);

            ref.reload(UnaryTestsF16::REF_UTINV_DPO_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPB_F16_ID,mgr);
         break;

         case TEST_SOLVE_LOWER_TRIANGULAR_F16_10:
            input1.reload(UnaryTestsF16::INPUT_LT_DPO_F16_ID,mgr);
            dims.reload(UnaryTestsF16::DIMSCHOLESKY1_DPO_S16_ID,mgr);
            input2.reload(UnaryTestsF16::INPUT_RNDA_DPO_F16_ID,mgr);

            ref.reload(UnaryTestsF16::REF_LTINV_DPO_F16_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF16::OUT_F16_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPA_F16_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF16::TMPB_F16_ID,mgr);
         break;
      }
       

    
    }

    void UnaryTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       //output.dump(mgr);
       (void)mgr;
    }
