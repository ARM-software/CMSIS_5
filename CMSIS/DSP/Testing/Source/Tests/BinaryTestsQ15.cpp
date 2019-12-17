#include "BinaryTestsQ15.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 70
#define SNR_LOW_THRESHOLD 30

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_HIGH_ERROR_Q15 ((q15_t)2000)
#define ABS_ERROR_Q15 ((q15_t)1000)

#define ABS_ERROR_Q63 ((q63_t)(1<<16))

#define MULT_SNR_THRESHOLD 60

#define ONEHALF 0x4000

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 40


#define LOADDATA2()                         \
      const q15_t *inp1=input1.ptr();       \
      const q15_t *inp2=input2.ptr();       \
                                            \
      q15_t *ap=a.ptr();                    \
      q15_t *bp=b.ptr();                    \
                                            \
      q15_t *outp=output.ptr();             \
      q15_t *tmpPtr=tmp.ptr();              \
      int16_t *dimsp = dims.ptr();          \
      int nbMatrixes = dims.nbSamples() / 3;\
      int rows,internal,columns;            \
      int i;


#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=internal;                                               \
      memcpy((void*)ap,(const void*)inp1,2*sizeof(q15_t)*rows*internal);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=internal;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,2*sizeof(q15_t)*internal*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;


    void BinaryTestsQ15::test_mat_mult_q15()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          arm_mat_mult_q15(&this->in1,&this->in2,&this->out,tmpPtr);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q15_t)SNR_LOW_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_HIGH_ERROR_Q15);

    } 

    void BinaryTestsQ15::test_mat_cmplx_mult_q15()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;


          PREPAREDATA2();

          arm_mat_cmplx_mult_q15(&this->in1,&this->in2,&this->out,tmpPtr);

          outp += (2*rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(q15_t)MULT_SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 


    void BinaryTestsQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


    
      switch(id)
      {
         case TEST_MAT_MULT_Q15_1:
            input1.reload(BinaryTestsQ15::INPUTS1_Q15_ID,mgr);
            input2.reload(BinaryTestsQ15::INPUTS2_Q15_ID,mgr);
            dims.reload(BinaryTestsQ15::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsQ15::REFMUL1_Q15_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ15::TMPA_Q15_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ15::TMPB_Q15_ID,mgr);
            tmp.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ15::TMP_Q15_ID,mgr);
         break;

         case TEST_MAT_CMPLX_MULT_Q15_2:
            input1.reload(BinaryTestsQ15::INPUTSC1_Q15_ID,mgr);
            input2.reload(BinaryTestsQ15::INPUTSC2_Q15_ID,mgr);
            dims.reload(BinaryTestsQ15::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsQ15::REFCMPLXMUL1_Q15_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsQ15::OUT_Q15_ID,mgr);
            a.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ15::TMPA_Q15_ID,mgr);
            b.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ15::TMPB_Q15_ID,mgr);
            tmp.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ15::TMP_Q15_ID,mgr);
         break;

    
      }
       

    
    }

    void BinaryTestsQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
