#include "arm_math.h"
#include "UnaryTestsF64.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)
#define ABS_ERROR (1.0e-5)

#define LOADDATA2()                          \
      const float64_t *inp1=input1.ptr();    \
      const float64_t *inp2=input2.ptr();    \
                                             \
      float64_t *ap=a.ptr();                 \
      float64_t *bp=b.ptr();                 \
                                             \
      float64_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define LOADDATA1()                          \
      const float64_t *inp1=input1.ptr();    \
                                             \
      float64_t *ap=a.ptr();                 \
                                             \
      float64_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float64_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=rows;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,sizeof(float64_t)*rows*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

#define PREPAREDATA1(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float64_t)*rows*columns);\
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




void UnaryTestsF64::test_mat_inverse_f64()
    {     
      const float64_t *inp1=input1.ptr();    
                                             
      float64_t *ap=a.ptr();                 
                                             
      float64_t *outp=output.ptr();          
      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();
      int rows,columns;                      
      int i;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATA1(false);

          arm_mat_inverse_f64(&this->in1,&this->out);

          outp += (rows * columns);
          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    }

    void UnaryTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
    
      switch(id)
      {
         case TEST_MAT_INVERSE_F64_5:
            input1.reload(UnaryTestsF64::INPUTSINV_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSINVERT1_S16_ID,mgr);

            ref.reload(UnaryTestsF64::REFINV1_F64_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF64::OUT_F64_ID,mgr);
            a.create(ref.nbSamples(),UnaryTestsF64::TMPA_F64_ID,mgr);
         break;
      }
       

    
    }

    void UnaryTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
