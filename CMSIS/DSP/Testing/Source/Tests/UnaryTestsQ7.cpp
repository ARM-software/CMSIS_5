#include "UnaryTestsQ7.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 20
#define SNR_LOW_THRESHOLD 11

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q7 ((q7_t)2)
#define ABS_ERROR_Q63 ((q63_t)(1<<16))

#define ONEHALF 0x4000

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 47

static void checkInnerTail(q7_t *b)
{
    ASSERT_TRUE(b[0] == 0);
    ASSERT_TRUE(b[1] == 0);
    ASSERT_TRUE(b[2] == 0);
    ASSERT_TRUE(b[3] == 0);
    ASSERT_TRUE(b[4] == 0);
    ASSERT_TRUE(b[5] == 0);
    ASSERT_TRUE(b[6] == 0);
    ASSERT_TRUE(b[7] == 0);

    ASSERT_TRUE(b[8] == 0);
    ASSERT_TRUE(b[9] == 0);
    ASSERT_TRUE(b[10] == 0);
    ASSERT_TRUE(b[11] == 0);
    ASSERT_TRUE(b[12] == 0);
    ASSERT_TRUE(b[13] == 0);
    ASSERT_TRUE(b[14] == 0);
    ASSERT_TRUE(b[15] == 0);
}

#define LOADDATA2()                          \
      const q7_t *inp1=input1.ptr();    \
      const q7_t *inp2=input2.ptr();    \
                                             \
      q7_t *ap=a.ptr();                 \
      q7_t *bp=b.ptr();                 \
                                             \
      q7_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define LOADDATA1()                          \
      const q7_t *inp1=input1.ptr();    \
                                             \
      q7_t *ap=a.ptr();                 \
                                             \
      q7_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(q7_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=rows;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,sizeof(q7_t)*rows*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

#define PREPAREDATA1(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(q7_t)*rows*columns);\
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
      const q7_t *inp1=input1.ptr();    \
      const q7_t *inp2=input2.ptr();    \
                                             \
      q7_t *ap=a.ptr();                 \
      q7_t *bp=b.ptr();                 \
                                             \
      q7_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() / 2;\
      int rows,internal;                      \
      int i;

#define PREPAREVECDATA2()                                            \
      in1.numRows=rows;                                              \
      in1.numCols=internal;                                          \
      memcpy((void*)ap,(const void*)inp1,sizeof(q7_t)*rows*internal);\
      in1.pData = ap;                                                \
                                                                     \
      memcpy((void*)bp,(const void*)inp2,sizeof(q7_t)*internal);

  void UnaryTestsQ7::test_mat_vec_mult_q7()
    {     
      LOADVECDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;

          PREPAREVECDATA2();

          arm_mat_vec_mult_q7(&this->in1, bp, outp);

          outp += rows ;
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q7_t)SNR_LOW_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);

    } 

void UnaryTestsQ7::test_mat_trans_q7()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;
          PREPAREDATA1(true);

          status=arm_mat_trans_q7(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q7_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);

    } 


    void UnaryTestsQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


      (void)params;
      switch(id)
      {
        
         case TEST_MAT_TRANS_Q7_1:
            input1.reload(UnaryTestsQ7::INPUTS1_Q7_ID,mgr);
            dims.reload(UnaryTestsQ7::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ7::REFTRANS1_Q7_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ7::OUT_Q7_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ7::TMPA_Q7_ID,mgr);
         break;

        
      

       case TEST_MAT_VEC_MULT_Q7_2:
            input1.reload(UnaryTestsQ7::INPUTS1_Q7_ID,mgr);
            input2.reload(UnaryTestsQ7::INPUTVEC1_Q7_ID,mgr);
            dims.reload(UnaryTestsQ7::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsQ7::REFVECMUL1_Q7_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsQ7::OUT_Q7_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsQ7::TMPA_Q7_ID,mgr);
            b.create(MAXMATRIXDIM,UnaryTestsQ7::TMPB_Q7_ID,mgr);
         break;
       }

    
    }

    void UnaryTestsQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
