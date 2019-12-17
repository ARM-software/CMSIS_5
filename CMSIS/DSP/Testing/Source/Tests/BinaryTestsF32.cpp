#include "BinaryTestsF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)
#define ABS_ERROR (1.0e-5)

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
      int nbMatrixes = dims.nbSamples() / 3;\
      int rows,internal,columns;                      \
      int i;


#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=internal;                                               \
      memcpy((void*)ap,(const void*)inp1,2*sizeof(float32_t)*rows*internal);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=internal;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,2*sizeof(float32_t)*internal*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;


    void BinaryTestsF32::test_mat_mult_f32()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          arm_mat_mult_f32(&this->in1,&this->in2,&this->out);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void BinaryTestsF32::test_mat_cmplx_mult_f32()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;


          PREPAREDATA2();

          arm_mat_cmplx_mult_f32(&this->in1,&this->in2,&this->out);

          outp += (2*rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 


    void BinaryTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


    
      switch(id)
      {
         case TEST_MAT_MULT_F32_1:
            input1.reload(BinaryTestsF32::INPUTS1_F32_ID,mgr);
            input2.reload(BinaryTestsF32::INPUTS2_F32_ID,mgr);
            dims.reload(BinaryTestsF32::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsF32::REFMUL1_F32_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF32::TMPB_F32_ID,mgr);
         break;

         case TEST_MAT_CMPLX_MULT_F32_2:
            input1.reload(BinaryTestsF32::INPUTSC1_F32_ID,mgr);
            input2.reload(BinaryTestsF32::INPUTSC2_F32_ID,mgr);
            dims.reload(BinaryTestsF32::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsF32::REFCMPLXMUL1_F32_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsF32::OUT_F32_ID,mgr);
            a.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF32::TMPA_F32_ID,mgr);
            b.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF32::TMPB_F32_ID,mgr);
         break;

    
      }
       

    
    }

    void BinaryTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
