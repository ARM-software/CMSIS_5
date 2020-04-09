#include "UnaryTestsF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)
#define ABS_ERROR (1.0e-5)

/*

Comparisons for inverse

*/

/* Not very accurate for big matrix.
But big matrix needed for checking the vectorized code */

#define SNR_THRESHOLD_INV 70
#define REL_ERROR_INV (1.0e-3)
#define ABS_ERROR_INV (1.0e-3)

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 40

#define LOADDATA2()                          \
      const float32_t *inp1=input1.ptr();    \
      const float32_t *inp2=input2.ptr();    \
                                             \
      float32_t *ap=a.ptr();                 \
      float32_t *bp=b.ptr();                 \
                                             \
      float32_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define LOADDATA1()                          \
      const float32_t *inp1=input1.ptr();    \
                                             \
      float32_t *ap=a.ptr();                 \
                                             \
      float32_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float32_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=rows;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,sizeof(float32_t)*rows*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

#define PREPAREDATA1(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float32_t)*rows*columns);\
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


    void UnaryTestsF32::test_mat_add_f32()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          arm_mat_add_f32(&this->in1,&this->in2,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_sub_f32()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          arm_mat_sub_f32(&this->in1,&this->in2,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_scale_f32()
    {     
      LOADDATA1();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(false);

          arm_mat_scale_f32(&this->in1,0.5f,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_trans_f32()
    {     
      LOADDATA1();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(true);

          arm_mat_trans_f32(&this->in1,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_inverse_f32()
    {     
      const float32_t *inp1=input1.ptr();    
                                             
      float32_t *ap=a.ptr();                 
                                             
      float32_t *outp=output.ptr();          
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

          status=arm_mat_inverse_f32(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD_INV);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR_INV,REL_ERROR_INV);

    }

    void UnaryTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


    
      switch(id)
      {
         case TEST_MAT_ADD_F32_1:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            input2.reload(UnaryTestsF32::INPUTS2_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFADD1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F32_ID,mgr);
         break;

         case TEST_MAT_SUB_F32_2:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            input2.reload(UnaryTestsF32::INPUTS2_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFSUB1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F32_ID,mgr);
         break;

         case TEST_MAT_SCALE_F32_3:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFSCALE1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
         break;

         case TEST_MAT_TRANS_F32_4:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFTRANS1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
         break;

         case TEST_MAT_INVERSE_F32_5:
            input1.reload(UnaryTestsF32::INPUTSINV_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSINVERT1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFINV1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
         break;
      }
       

    
    }

    void UnaryTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
