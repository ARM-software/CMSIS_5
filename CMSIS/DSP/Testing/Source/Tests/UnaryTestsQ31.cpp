#include "UnaryTestsQ31.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 100

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)2)
#define ABS_ERROR_Q63 ((q63_t)(1<<16))

#define ONEHALF 0x40000000

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 40

static void checkInnerTail(q31_t *b)
{
    ASSERT_TRUE(b[0] == 0);
    ASSERT_TRUE(b[1] == 0);
    ASSERT_TRUE(b[2] == 0);
    ASSERT_TRUE(b[3] == 0);
}


#define LOADDATA2()                          \
      const q31_t *inp1=input1.ptr();    \
      const q31_t *inp2=input2.ptr();    \
                                             \
      q31_t *ap=a.ptr();                 \
      q31_t *bp=b.ptr();                 \
                                             \
      q31_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define LOADDATA1()                          \
      const q31_t *inp1=input1.ptr();    \
                                             \
      q31_t *ap=a.ptr();                 \
                                             \
      q31_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(q31_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=rows;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,sizeof(q31_t)*rows*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

#define PREPAREDATA1(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(q31_t)*rows*columns);\
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
      memcpy((void*)ap,(const void*)inp1,2*sizeof(q31_t)*rows*columns);\
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
      const q31_t *inp1=input1.ptr();    \
      const q31_t *inp2=input2.ptr();    \
                                             \
      q31_t *ap=a.ptr();                 \
      q31_t *bp=b.ptr();                 \
                                             \
      q31_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() / 2;\
      int rows,internal;                      \
      int i;

#define PREPAREVECDATA2()                                             \
      in1.numRows=rows;                                               \
      in1.numCols=internal;                                           \
      memcpy((void*)ap,(const void*)inp1,sizeof(q31_t)*rows*internal);\
      in1.pData = ap;                                                 \
                                                                      \
      memcpy((void*)bp,(const void*)inp2,sizeof(q31_t)*internal);


    void UnaryTestsQ31::test_mat_vec_mult_q31()
    {     
      LOADVECDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;

          PREPAREVECDATA2();

          arm_mat_vec_mult_q31(&this->in1, bp, outp);

          outp += rows ;
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

    void UnaryTestsQ31::test_mat_add_q31()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_add_q31(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

void UnaryTestsQ31::test_mat_sub_q31()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_sub_q31(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

void UnaryTestsQ31::test_mat_scale_q31()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(false);

          status=arm_mat_scale_q31(&this->in1,ONEHALF,0,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

void UnaryTestsQ31::test_mat_trans_q31()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(true);

          status=arm_mat_trans_q31(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

void UnaryTestsQ31::test_mat_cmplx_trans_q31()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1C(true);

          status=arm_mat_cmplx_trans_q31(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += 2*(rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 


    void UnaryTestsQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


      (void)params;
      switch(id)
      {
         case TEST_MAT_ADD_Q31_1:
            input1.reload(UnaryTestsQ31::INPUTS1_Q31_ID,mgr);
            input2.reload(UnaryTestsQ31::INPUTS2_Q31_ID,mgr);
            dims.reload(UnaryTestsQ31::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ31::REFADD1_Q31_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPA_Q31_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPB_Q31_ID,mgr);
         break;

         case TEST_MAT_SUB_Q31_2:
            input1.reload(UnaryTestsQ31::INPUTS1_Q31_ID,mgr);
            input2.reload(UnaryTestsQ31::INPUTS2_Q31_ID,mgr);
            dims.reload(UnaryTestsQ31::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ31::REFSUB1_Q31_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPA_Q31_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPB_Q31_ID,mgr);
         break;

         case TEST_MAT_SCALE_Q31_3:
            input1.reload(UnaryTestsQ31::INPUTS1_Q31_ID,mgr);
            dims.reload(UnaryTestsQ31::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ31::REFSCALE1_Q31_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPA_Q31_ID,mgr);
         break;

         case TEST_MAT_TRANS_Q31_4:
            input1.reload(UnaryTestsQ31::INPUTS1_Q31_ID,mgr);
            dims.reload(UnaryTestsQ31::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ31::REFTRANS1_Q31_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPA_Q31_ID,mgr);
         break;

         case TEST_MAT_VEC_MULT_Q31_5:
            input1.reload(UnaryTestsQ31::INPUTS1_Q31_ID,mgr);
            input2.reload(UnaryTestsQ31::INPUTVEC1_Q31_ID,mgr);
            dims.reload(UnaryTestsQ31::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ31::REFVECMUL1_Q31_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPA_Q31_ID,mgr);
            b.create(MAXMATRIXDIM,UnaryTestsQ31::TMPB_Q31_ID,mgr);
         break;

         case TEST_MAT_CMPLX_TRANS_Q31_6:
            input1.reload(UnaryTestsQ31::INPUTSC1_Q31_ID,mgr);
            dims.reload(UnaryTestsQ31::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ31::REFTRANSC1_Q31_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ31::TMPA_Q31_ID,mgr);
         break;

        
      }
       

    
    }

    void UnaryTestsQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
