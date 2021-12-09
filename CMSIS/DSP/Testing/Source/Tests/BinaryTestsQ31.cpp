#include "BinaryTestsQ31.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 100

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)5)
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
      int nbMatrixes = dims.nbSamples() / 3;\
      int rows,internal,columns;                      \
      int i;


#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=internal;                                               \
      memcpy((void*)ap,(const void*)inp1,2*sizeof(q31_t)*rows*internal);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=internal;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,2*sizeof(q31_t)*internal*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;




    void BinaryTestsQ31::test_mat_mult_q31()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_mult_q31(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);

      }

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 



    void BinaryTestsQ31::test_mat_cmplx_mult_q31()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_cmplx_mult_q31(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);
        
          outp += (2*rows * columns);
          checkInnerTail(outp);
      }

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

    void BinaryTestsQ31::test_mat_mult_opt_q31()
    {     
      LOADDATA2();
      q31_t *tmpPtr=tmp.ptr();      

      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();
          memset(tmpPtr,0,sizeof(q31_t)*internal*columns + 16);
          status=arm_mat_mult_opt_q31(&this->in1,&this->in2,&this->out,tmpPtr);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          checkInnerTail(outp);
          checkInnerTail(tmpPtr + internal*columns);

      }

      ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

      ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 


    void BinaryTestsQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


      (void)params;
      switch(id)
      {
         case TEST_MAT_MULT_Q31_1:
            input1.reload(BinaryTestsQ31::INPUTS1_Q31_ID,mgr);
            input2.reload(BinaryTestsQ31::INPUTS2_Q31_ID,mgr);
            dims.reload(BinaryTestsQ31::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsQ31::REFMUL1_Q31_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ31::TMPA_Q31_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ31::TMPB_Q31_ID,mgr);
         break;

         case TEST_MAT_CMPLX_MULT_Q31_2:
            input1.reload(BinaryTestsQ31::INPUTSC1_Q31_ID,mgr);
            input2.reload(BinaryTestsQ31::INPUTSC2_Q31_ID,mgr);
            dims.reload(BinaryTestsQ31::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsQ31::REFCMPLXMUL1_Q31_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ31::TMPA_Q31_ID,mgr);
            b.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ31::TMPB_Q31_ID,mgr);
         break;

         case TEST_MAT_MULT_OPT_Q31_3:
            input1.reload(BinaryTestsQ31::INPUTS1_Q31_ID,mgr);
            input2.reload(BinaryTestsQ31::INPUTS2_Q31_ID,mgr);
            dims.reload(BinaryTestsQ31::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsQ31::REFMUL1_Q31_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsQ31::OUT_Q31_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ31::TMPA_Q31_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ31::TMPB_Q31_ID,mgr);

            tmp.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsQ31::TMPC_Q31_ID,mgr);

         break;



    
      }
       

    
    }

    void BinaryTestsQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
