#include "UnaryTestsQ15.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 70

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)4)
#define ABS_ERROR_Q63 ((q63_t)(1<<16))

#define ONEHALF 0x4000

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 40

static void refInnerTail(q15_t *b)
{
    b[0] = 1;
    b[1] = -1;
    b[2] = 2;
    b[3] = -2;
    b[4] = 3;
    b[5] = -3;
    b[6] = 4;
    b[7] = -4;
}

static void checkInnerTail(q15_t *b)
{
    ASSERT_TRUE(b[0] == 1);
    ASSERT_TRUE(b[1] == -1);
    ASSERT_TRUE(b[2] == 2);
    ASSERT_TRUE(b[3] == -2);
    ASSERT_TRUE(b[4] == 3);
    ASSERT_TRUE(b[5] == -3);
    ASSERT_TRUE(b[6] == 4);
    ASSERT_TRUE(b[7] == -4);
}


#define LOADDATA2()                          \
      const q15_t *inp1=input1.ptr();    \
      const q15_t *inp2=input2.ptr();    \
                                             \
      q15_t *ap=a.ptr();                 \
      q15_t *bp=b.ptr();                 \
                                             \
      q15_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define LOADDATA1()                          \
      const q15_t *inp1=input1.ptr();    \
                                             \
      q15_t *ap=a.ptr();                 \
                                             \
      q15_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(q15_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=rows;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,sizeof(q15_t)*rows*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

#define PREPAREDATA1(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(q15_t)*rows*columns);\
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
      memcpy((void*)ap,(const void*)inp1,2*sizeof(q15_t)*rows*columns);\
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
      const q15_t *inp1=input1.ptr();    \
      const q15_t *inp2=input2.ptr();    \
                                             \
      q15_t *ap=a.ptr();                 \
      q15_t *bp=b.ptr();                 \
                                             \
      q15_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() / 2;\
      int rows,internal;                      \
      int i;

#define PREPAREVECDATA2()                                             \
      in1.numRows=rows;                                               \
      in1.numCols=internal;                                           \
      memcpy((void*)ap,(const void*)inp1,sizeof(q15_t)*rows*internal);\
      in1.pData = ap;                                                 \
                                                                      \
      memcpy((void*)bp,(const void*)inp2,sizeof(q15_t)*internal);


    void UnaryTestsQ15::test_mat_vec_mult_q15()
    {     


      LOADVECDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;

          PREPAREVECDATA2();
          refInnerTail(outp + rows);
          arm_mat_vec_mult_q15(&this->in1, bp, outp);

          outp += rows ;
          checkInnerTail(outp);

      }


      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

    void UnaryTestsQ15::test_mat_add_q15()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();
          refInnerTail(outp + rows * columns);
          status=arm_mat_add_q15(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }


      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

void UnaryTestsQ15::test_mat_sub_q15()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();
          refInnerTail(outp + rows * columns);
          status=arm_mat_sub_q15(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }


      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

void UnaryTestsQ15::test_mat_scale_q15()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(false);
          refInnerTail(outp + rows * columns);
          status=arm_mat_scale_q15(&this->in1,ONEHALF,0,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }


      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

void UnaryTestsQ15::test_mat_trans_q15()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(true);
          refInnerTail(outp + rows * columns);
          status=arm_mat_trans_q15(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

void UnaryTestsQ15::test_mat_cmplx_trans_q15()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1C(true);
          refInnerTail(outp + 2*rows * columns);
          status=arm_mat_cmplx_trans_q15(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += 2*(rows * columns);
          checkInnerTail(outp);

      }


      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    }


    void UnaryTestsQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


      (void)params;
      switch(id)
      {
         case TEST_MAT_ADD_Q15_1:
            input1.reload(UnaryTestsQ15::INPUTS1_Q15_ID,mgr);
            input2.reload(UnaryTestsQ15::INPUTS2_Q15_ID,mgr);
            dims.reload(UnaryTestsQ15::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ15::REFADD1_Q15_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPA_Q15_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPB_Q15_ID,mgr);
         break;

         case TEST_MAT_SUB_Q15_2:
            input1.reload(UnaryTestsQ15::INPUTS1_Q15_ID,mgr);
            input2.reload(UnaryTestsQ15::INPUTS2_Q15_ID,mgr);
            dims.reload(UnaryTestsQ15::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ15::REFSUB1_Q15_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPA_Q15_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPB_Q15_ID,mgr);
         break;

         case TEST_MAT_SCALE_Q15_3:
            input1.reload(UnaryTestsQ15::INPUTS1_Q15_ID,mgr);
            dims.reload(UnaryTestsQ15::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ15::REFSCALE1_Q15_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPA_Q15_ID,mgr);
         break;

         case TEST_MAT_TRANS_Q15_4:
            input1.reload(UnaryTestsQ15::INPUTS1_Q15_ID,mgr);
            dims.reload(UnaryTestsQ15::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ15::REFTRANS1_Q15_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPA_Q15_ID,mgr);
         break;

         case TEST_MAT_VEC_MULT_Q15_5:
            input1.reload(UnaryTestsQ15::INPUTS1_Q15_ID,mgr);
            input2.reload(UnaryTestsQ15::INPUTVEC1_Q15_ID,mgr);
            dims.reload(UnaryTestsQ15::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ15::REFVECMUL1_Q15_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPA_Q15_ID,mgr);
            b.create(MAXMATRIXDIM,UnaryTestsQ15::TMPB_Q15_ID,mgr);
         break;

         case TEST_MAT_CMPLX_TRANS_Q15_6:
            input1.reload(UnaryTestsQ15::INPUTSC1_Q15_ID,mgr);
            dims.reload(UnaryTestsQ15::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ15::REFTRANSC1_Q15_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ15::TMPA_Q15_ID,mgr);
         break;

        
      }
       

    
    }

    void UnaryTestsQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
