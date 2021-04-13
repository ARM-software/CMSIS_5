#include "BinaryTestsF64.h"
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
      const float64_t *inp1=input1.ptr();    \
      const float64_t *inp2=input2.ptr();    \
                                             \
      float64_t *ap=a.ptr();                 \
      float64_t *bp=b.ptr();                 \
                                             \
      float64_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() / 3;\
      int rows,internal,columns;                      \
      int i;





#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=internal;                                               \
      memcpy((void*)ap,(const void*)inp1,2*sizeof(float64_t)*rows*internal);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=internal;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,2*sizeof(float64_t)*internal*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

                                             

    void BinaryTestsF64::test_mat_mult_f64()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_mult_f64(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    
#if 0
    void BinaryTestsF64::test_mat_cmplx_mult_f64()
    {     
      LOADDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;
          columns = *dimsp++;


          PREPAREDATA2();

          arm_mat_cmplx_mult_f64(&this->in1,&this->in2,&this->out);

          outp += (2*rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

#endif
    void BinaryTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


      (void)params;
      switch(id)
      {
         case TEST_MAT_MULT_F64_1:
            input1.reload(BinaryTestsF64::INPUTS1_F64_ID,mgr);
            input2.reload(BinaryTestsF64::INPUTS2_F64_ID,mgr);
            dims.reload(BinaryTestsF64::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsF64::REFMUL1_F64_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsF64::OUT_F64_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF64::TMPA_F64_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF64::TMPB_F64_ID,mgr);
         break;
#if 0
         case TEST_MAT_CMPLX_MULT_F64_2:
            input1.reload(BinaryTestsF64::INPUTSC1_F64_ID,mgr);
            input2.reload(BinaryTestsF64::INPUTSC2_F64_ID,mgr);
            dims.reload(BinaryTestsF64::DIMSBINARY1_S16_ID,mgr);

            ref.reload(BinaryTestsF64::REFCMPLXMUL1_F64_ID,mgr);

            output.create(ref.nbSamples(),BinaryTestsF64::OUT_F64_ID,mgr);
            a.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF64::TMPA_F64_ID,mgr);
            b.create(2*MAXMATRIXDIM*MAXMATRIXDIM,BinaryTestsF64::TMPB_F64_ID,mgr);
         break;
#endif
         

    
      }
       

    
    }

    void BinaryTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
