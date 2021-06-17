#include "UnaryTestsF64.h"
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-6)
#define ABS_ERROR (1.0e-5)

/*

Comparison for Cholesky

*/
#define SNR_THRESHOLD_CHOL 270
#define REL_ERROR_CHOL (1.0e-9)
#define ABS_ERROR_CHOL (1.0e-9)

/* LDLT comparison */

#define REL_ERROR_LDLT (1e-5)
#define ABS_ERROR_LDLT (1e-5)

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

#define PREPAREDATALL1()                                                 \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float64_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      outll.numRows=rows;                                                \
      outll.numCols=columns;                                             \
                                                                         \
      outll.pData = outllp;

#define SWAP_ROWS(A,i,j)     \
  for(int w=0;w < n; w++)    \
  {                          \
     float64_t tmp;          \
     tmp = A[i*n + w];       \
     A[i*n + w] = A[j*n + w];\
     A[j*n + w] = tmp;       \
  }



void UnaryTestsF64::test_mat_add_f64()
{

}

void UnaryTestsF64::test_mat_sub_f64()
{
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_sub_f64(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);
}

void UnaryTestsF64::test_mat_scale_f64()
{

}

void UnaryTestsF64::test_mat_trans_f64()
{
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(true);

          status=arm_mat_trans_f64(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

}

/*

Test framework is only adding 16 bytes of free memory after the end of a buffer.
So, we limit to 2 float64 for checking out of buffer write.
*/
static void refInnerTail(float64_t *b)
{
    b[0] = 1.0;
    b[1] = -2.0;
}

static void checkInnerTail(float64_t *b)
{
    ASSERT_TRUE(b[0] == 1.0);
    ASSERT_TRUE(b[1] == -2.0);
}


void UnaryTestsF64::test_mat_inverse_f64()
    {     
      const float64_t *inp1=input1.ptr();    
                                             
      float64_t *ap=a.ptr();                 
                                             
      float64_t *outp=output.ptr();          
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

          refInnerTail(outp+(rows * columns));

          status=arm_mat_inverse_f64(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);

          checkInnerTail(outp);

      }


      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    }

    void UnaryTestsF64::test_mat_cholesky_dpo_f64()
    {
      float64_t *ap=a.ptr();                 
      const float64_t *inp1=input1.ptr();    
                                             
                                             
      float64_t *outp=output.ptr();     
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

          status=arm_mat_cholesky_f64(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD_CHOL);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR_CHOL,REL_ERROR_CHOL);
    }

    void UnaryTestsF64::test_solve_upper_triangular_f64()
    {
      float64_t *ap=a.ptr();                 
      const float64_t *inp1=input1.ptr();    

      float64_t *bp=b.ptr();                 
      const float64_t *inp2=input2.ptr();    
                                             
                                             
      float64_t *outp=output.ptr();     
      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();

      int rows,columns;                      
      int i;
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATA2();

          status=arm_mat_solve_upper_triangular_f64(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);
          inp2 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
    }

    void UnaryTestsF64::test_solve_lower_triangular_f64()
    {
      float64_t *ap=a.ptr();                 
      const float64_t *inp1=input1.ptr();    

      float64_t *bp=b.ptr();                 
      const float64_t *inp2=input2.ptr();    
                                             
                                             
      float64_t *outp=output.ptr();     
      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();

      int rows,columns;                      
      int i;
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATA2();

          status=arm_mat_solve_lower_triangular_f64(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);
          inp2 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
    }

    static void trans_f64(const float64_t *src, float64_t *dst, int n)
    {
        for(int r=0; r<n ; r++)
        {
          for(int c=0; c<n ; c++)
          {
              dst[c*n+r] = src[r*n+c];
          }
        }
    }

    static void mult_f64_f64(const float64_t *srcA, const float64_t *srcB, float64_t *dst,int n)
    {
        for(int r=0; r<n ; r++)
        {
          for(int c=0; c<n ; c++)
          {
             float64_t sum=0.0;
             for(int k=0; k < n ; k++)
             {
                sum += srcA[r*n+k] * srcB[k*n+c];
             }
             dst[r*n+c] = sum;
          }
        }
    }

    void UnaryTestsF64::compute_ldlt_error(const int n,const int16_t *outpp)
    {
           float64_t *tmpa  = tmpapat.ptr() ;
           float64_t *tmpb  = tmpbpat.ptr() ;
           float64_t *tmpc  = tmpcpat.ptr() ;
                                           
          
          /* Compute P A P^t */

          // Create identiy matrix
          for(int r=0; r < n; r++)
          {
            for(int c=0; c < n; c++)
            {
               if (r == c)
               {
                 tmpa[r*n+c] = 1.0;
               }
               else
               {
                 tmpa[r*n+c] = 0.0;
               }
            }
          }

          // Create permutation matrix

          for(int r=0;r < n; r++)
          {
            SWAP_ROWS(tmpa,r,outpp[r]);
          }

          

          trans_f64((const float64_t*)tmpa,tmpb,n);
          mult_f64_f64((const float64_t*)this->in1.pData,(const float64_t*)tmpb,tmpc,n);
          mult_f64_f64((const float64_t*)tmpa,(const float64_t*)tmpc,outa,n);


          /* Compute L D L^t */
          trans_f64((const float64_t*)this->outll.pData,tmpc,n);
          mult_f64_f64((const float64_t*)this->outd.pData,(const float64_t*)tmpc,tmpa,n);
          mult_f64_f64((const float64_t*)this->outll.pData,(const float64_t*)tmpa,outb,n);

          
    }

    void UnaryTestsF64::test_mat_ldl_f64()
    {
      float64_t *ap=a.ptr();                 
      const float64_t *inp1=input1.ptr();  

                                            
      float64_t *outllp=outputll.ptr();   
      float64_t *outdp=outputd.ptr();   
      int16_t *outpp=outputp.ptr();   


      outa=outputa.ptr();   
      outb=outputb.ptr();   

      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();

      int rows,columns;                      
      int i;
      arm_status status;


      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATALL1();

          outd.numRows=rows;
          outd.numCols=columns;
          outd.pData=outdp;

          


          memset(outpp,0,rows*sizeof(uint16_t));
          memset(outdp,0,columns*rows*sizeof(float64_t));

          status=arm_mat_ldlt_f64(&this->in1,&this->outll,&this->outd,(uint16_t*)outpp);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          compute_ldlt_error(rows,outpp);

          outllp += (rows * columns);
          outdp += (rows * columns);
          outpp += rows;

          outa += (rows * columns);
          outb +=(rows * columns);

          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(outputll);
      ASSERT_EMPTY_TAIL(outputd);
      ASSERT_EMPTY_TAIL(outputp);
      ASSERT_EMPTY_TAIL(outputa);
      ASSERT_EMPTY_TAIL(outputb);


      ASSERT_CLOSE_ERROR(outputa,outputb,ABS_ERROR_LDLT,REL_ERROR_LDLT);


     
    

    }


    void UnaryTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
    
      (void)params;
      switch(id)
      {
          case TEST_MAT_SUB_F64_2:
            input1.reload(UnaryTestsF64::INPUTS1_F64_ID,mgr);
            input2.reload(UnaryTestsF64::INPUTS2_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF64::REFSUB1_F64_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF64::OUT_F64_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPA_F64_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPB_F64_ID,mgr);
          break;
          case TEST_MAT_TRANS_F64_4:
            input1.reload(UnaryTestsF64::INPUTS1_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF64::REFTRANS1_F64_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF64::OUT_F64_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPA_F64_ID,mgr);
         break;
         
         case TEST_MAT_INVERSE_F64_5:
            input1.reload(UnaryTestsF64::INPUTSINV_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSINVERT1_S16_ID,mgr);

            ref.reload(UnaryTestsF64::REFINV1_F64_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF64::OUT_F64_ID,mgr);
            a.create(ref.nbSamples(),UnaryTestsF64::TMPA_F64_ID,mgr);
         break;

         case TEST_MAT_CHOLESKY_DPO_F64_6:
            input1.reload(UnaryTestsF64::INPUTSCHOLESKY1_DPO_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSCHOLESKY1_DPO_S16_ID,mgr);

            ref.reload(UnaryTestsF64::REFCHOLESKY1_DPO_F64_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF64::OUT_F64_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPA_F64_ID,mgr);
         break;

         case TEST_SOLVE_UPPER_TRIANGULAR_F64_7:
            input1.reload(UnaryTestsF64::INPUT_UT_DPO_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSCHOLESKY1_DPO_S16_ID,mgr);
            input2.reload(UnaryTestsF64::INPUT_RNDA_DPO_F64_ID,mgr);

            ref.reload(UnaryTestsF64::REF_UTINV_DPO_F64_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF64::OUT_F64_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPA_F64_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPB_F64_ID,mgr);
         break;

         case TEST_SOLVE_LOWER_TRIANGULAR_F64_8:
            input1.reload(UnaryTestsF64::INPUT_LT_DPO_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSCHOLESKY1_DPO_S16_ID,mgr);
            input2.reload(UnaryTestsF64::INPUT_RNDA_DPO_F64_ID,mgr);

            ref.reload(UnaryTestsF64::REF_LTINV_DPO_F64_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF64::OUT_F64_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPA_F64_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPB_F64_ID,mgr);
         break;

         case TEST_MAT_LDL_F64_9:
            // Definite positive test
            input1.reload(UnaryTestsF64::INPUTSCHOLESKY1_DPO_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSCHOLESKY1_DPO_S16_ID,mgr);

            outputll.create(input1.nbSamples(),UnaryTestsF64::LL_F64_ID,mgr);
            outputd.create(input1.nbSamples(),UnaryTestsF64::D_F64_ID,mgr);
            outputp.create(input1.nbSamples(),UnaryTestsF64::PERM_S16_ID,mgr);

            outputa.create(input1.nbSamples(),UnaryTestsF64::OUTA_F64_ID,mgr);
            outputb.create(input1.nbSamples(),UnaryTestsF64::OUTA_F64_ID,mgr);

            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPA_F64_ID,mgr);
            
            tmpapat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPDB_F64_ID,mgr);
            tmpbpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPDC_F64_ID,mgr);
            tmpcpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPDD_F64_ID,mgr);

         break;

         case TEST_MAT_LDL_F64_10:
            // Semi definite positive test
            input1.reload(UnaryTestsF64::INPUTSCHOLESKY1_SDPO_F64_ID,mgr);
            dims.reload(UnaryTestsF64::DIMSCHOLESKY1_SDPO_S16_ID,mgr);
           
            outputll.create(input1.nbSamples(),UnaryTestsF64::LL_F64_ID,mgr);
            outputd.create(input1.nbSamples(),UnaryTestsF64::D_F64_ID,mgr);
            outputp.create(input1.nbSamples(),UnaryTestsF64::PERM_S16_ID,mgr);

            outputa.create(input1.nbSamples(),UnaryTestsF64::OUTA_F64_ID,mgr);
            outputb.create(input1.nbSamples(),UnaryTestsF64::OUTA_F64_ID,mgr);

            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPA_F64_ID,mgr);
            
            tmpapat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPDB_F64_ID,mgr);
            tmpbpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPDC_F64_ID,mgr);
            tmpcpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF64::TMPDD_F64_ID,mgr);

         break;

      }
       

    
    }

    void UnaryTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       //output.dump(mgr);
       (void)mgr;
    }
