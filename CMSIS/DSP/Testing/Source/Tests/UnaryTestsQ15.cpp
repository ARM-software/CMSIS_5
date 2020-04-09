#include "UnaryTestsQ15.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 70

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)2)
#define ABS_ERROR_Q63 ((q63_t)(1<<16))

#define ONEHALF 0x4000

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 40

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


    void UnaryTestsQ15::test_mat_add_q15()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          arm_mat_add_q15(&this->in1,&this->in2,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

void UnaryTestsQ15::test_mat_sub_q15()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          arm_mat_sub_q15(&this->in1,&this->in2,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

void UnaryTestsQ15::test_mat_scale_q15()
    {     
      LOADDATA1();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(false);

          arm_mat_scale_q15(&this->in1,ONEHALF,0,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

void UnaryTestsQ15::test_mat_trans_q15()
    {     
      LOADDATA1();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(true);

          arm_mat_trans_q15(&this->in1,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 


    void UnaryTestsQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


    
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

        
      }
       

    
    }

    void UnaryTestsQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
