#include "UnaryTestsF32.h"
#include "Error.h"


#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.0e-5)
#define ABS_ERROR (1.0e-5)

/*

Comparisons for inverse

*/

/* Not very accurate for big matrix.
But big matrix needed for checking the vectorized code */

#define SNR_THRESHOLD_INV 67
#define REL_ERROR_INV (1.0e-3)
#define ABS_ERROR_INV (1.0e-3)

/*

Comparison for Cholesky

*/
#define SNR_THRESHOLD_CHOL 92
#define REL_ERROR_CHOL (1.0e-5)
#define ABS_ERROR_CHOL (5.0e-4)

/* LDLT comparison */

#define REL_ERROR_LDLT (1e-5)
#define ABS_ERROR_LDLT (1e-5)

#define REL_ERROR_LDLT_SPDO (1e-5)
#define ABS_ERROR_LDLT_SDPO (2e-1)

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
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define LOADDATA1()                          \
      const float32_t *inp1=input1.ptr();    \
                                             \
      float32_t *ap=a.ptr();                 \
                                             \
      float32_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() >> 1;\
      int rows,columns;                      \
      int i;

#define PREPAREDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float32_t)*rows*columns);\
      in1.pData = ap;                                                    \
                                                                         \
      in2.numRows=rows;                                                  \
      in2.numCols=columns;                                               \
      memcpy((void*)bp,(const void*)inp2,sizeof(float32_t)*rows*columns);\
      in2.pData = bp;                                                    \
                                                                         \
      out.numRows=rows;                                                  \
      out.numCols=columns;                                               \
      out.pData = outp;

#define PREPAREDATA1(TRANSPOSED)                                         \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float32_t)*rows*columns);\
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
      memcpy((void*)ap,(const void*)inp1,2*sizeof(float32_t)*rows*columns);\
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
      const float32_t *inp1=input1.ptr();    \
      const float32_t *inp2=input2.ptr();    \
                                             \
      float32_t *ap=a.ptr();                 \
      float32_t *bp=b.ptr();                 \
                                             \
      float32_t *outp=output.ptr();          \
      int16_t *dimsp = dims.ptr();           \
      int nbMatrixes = dims.nbSamples() / 2;\
      int rows,internal;                      \
      int i;

#define PREPAREVECDATA2()                                                   \
      in1.numRows=rows;                                                  \
      in1.numCols=internal;                                               \
      memcpy((void*)ap,(const void*)inp1,2*sizeof(float32_t)*rows*internal);\
      in1.pData = ap;                                                    \
                                                                         \
      memcpy((void*)bp,(const void*)inp2,2*sizeof(float32_t)*internal);
                            
#define PREPAREDATALL1()                                                 \
      in1.numRows=rows;                                                  \
      in1.numCols=columns;                                               \
      memcpy((void*)ap,(const void*)inp1,sizeof(float32_t)*rows*columns);\
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


void UnaryTestsF32::test_mat_vec_mult_f32()
    {     
      LOADVECDATA2();

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          internal = *dimsp++;

          PREPAREVECDATA2();

          arm_mat_vec_mult_f32(&this->in1, bp, outp);

          outp += rows ;

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

    void UnaryTestsF32::test_mat_add_f32()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_add_f32(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_sub_f32()
    {     
      LOADDATA2();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA2();

          status=arm_mat_sub_f32(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_scale_f32()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(false);

          status=arm_mat_scale_f32(&this->in1,0.5f,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_trans_f32()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1(true);

          status=arm_mat_trans_f32(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

void UnaryTestsF32::test_mat_cmplx_trans_f32()
    {     
      LOADDATA1();
      arm_status status;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = *dimsp++;

          PREPAREDATA1C(true);

          status=arm_mat_cmplx_trans_f32(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += 2*(rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    }

void UnaryTestsF32::test_mat_inverse_f32()
    {     
      const float32_t *inp1=input1.ptr();    
                                             
      float32_t *ap=a.ptr();                 
                                             
      float32_t *outp=output.ptr();          
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

          status=arm_mat_inverse_f32(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD_INV);

      ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR_INV,REL_ERROR_INV);

    }

    void UnaryTestsF32::test_mat_cholesky_dpo_f32()
    {
      float32_t *ap=a.ptr();                 
      const float32_t *inp1=input1.ptr();    
                                             
                                             
      float32_t *outp=output.ptr();     
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

          status=arm_mat_cholesky_f32(&this->in1,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD_CHOL);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR_CHOL,REL_ERROR_CHOL);
    }

    void UnaryTestsF32::test_solve_upper_triangular_f32()
    {
      float32_t *ap=a.ptr();                 
      const float32_t *inp1=input1.ptr();    

      float32_t *bp=b.ptr();                 
      const float32_t *inp2=input2.ptr();    
                                             
                                             
      float32_t *outp=output.ptr();     
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

          status=arm_mat_solve_upper_triangular_f32(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);
          inp2 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

      ASSERT_CLOSE_ERROR(ref,output,ABS_ERROR,REL_ERROR);
    }

    void UnaryTestsF32::test_solve_lower_triangular_f32()
    {
      float32_t *ap=a.ptr();                 
      const float32_t *inp1=input1.ptr();    

      float32_t *bp=b.ptr();                 
      const float32_t *inp2=input2.ptr();    
                                             
                                             
      float32_t *outp=output.ptr();     
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

          status=arm_mat_solve_lower_triangular_f32(&this->in1,&this->in2,&this->out);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);

          outp += (rows * columns);
          inp1 += (rows * columns);
          inp2 += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(output);

      ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

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

    static void trans_f32_f64(const float32_t *src, float64_t *dst, int n)
    {
        for(int r=0; r<n ; r++)
        {
          for(int c=0; c<n ; c++)
          {
             dst[c*n+r] = (float64_t)src[r*n+c];
          }
        }
    }

    static void mult_f32_f64(const float32_t *srcA, const float64_t *srcB, float64_t *dst,int n)
    {
        for(int r=0; r<n ; r++)
        {
          for(int c=0; c<n ; c++)
          {
             float64_t sum=0.0;
             for(int k=0; k < n ; k++)
             {
                sum += (float64_t)srcA[r*n+k] * srcB[k*n+c];
             }
             dst[r*n+c] = sum;
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
    
    void UnaryTestsF32::compute_ldlt_error(const int n,const int16_t *outpp)
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
          mult_f32_f64((const float32_t*)this->in1.pData,(const float64_t*)tmpb,tmpc,n);
          mult_f64_f64((const float64_t*)tmpa,(const float64_t*)tmpc,outa,n);
         


          /* Compute L D L^t */
          trans_f32_f64((const float32_t*)this->outll.pData,tmpc,n);
          mult_f32_f64((const float32_t*)this->outd.pData,(const float64_t*)tmpc,tmpa,n);
          mult_f32_f64((const float32_t*)this->outll.pData,(const float64_t*)tmpa,outb,n);

         
          
    }


    void UnaryTestsF32::test_mat_ldl_f32()
    {
      float32_t *ap=a.ptr();                 
      const float32_t *inp1=input1.ptr();  

                                        
      float32_t *outllp=outputll.ptr();   
      float32_t *outdp=outputd.ptr();   
      int16_t *outpp=outputp.ptr();   


      outa=outputa.ptr();   
      outb=outputb.ptr();   

      int16_t *dimsp = dims.ptr();           
      int nbMatrixes = dims.nbSamples();

      int rows,columns;                      
      int i;
      arm_status status;

      int nb=0;

      for(i=0;i < nbMatrixes ; i ++)
      {
          rows = *dimsp++;
          columns = rows;

          PREPAREDATALL1();

          outd.numRows=rows;
          outd.numCols=columns;
          outd.pData=outdp;

          memset(outpp,0,rows*sizeof(uint16_t));
          memset(outdp,0,columns*rows*sizeof(float32_t));

          status=arm_mat_ldlt_f32(&this->in1,&this->outll,&this->outd,(uint16_t*)outpp);
          ASSERT_TRUE(status==ARM_MATH_SUCCESS);
 

          compute_ldlt_error(rows,outpp);

         
          outllp += (rows * columns);
          outdp += (rows * columns);
          outpp += rows;

          outa += (rows * columns);
          outb +=(rows * columns);

          inp1 += (rows * columns);

          nb += (rows * columns);

      }

      ASSERT_EMPTY_TAIL(outputll);
      ASSERT_EMPTY_TAIL(outputd);
      ASSERT_EMPTY_TAIL(outputp);
      ASSERT_EMPTY_TAIL(outputa);
      ASSERT_EMPTY_TAIL(outputb);


      ASSERT_CLOSE_ERROR(outputa,outputb,snrAbs,snrRel);


  
    }

    void UnaryTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


      (void)params;
      switch(id)
      {
         case TEST_MAT_ADD_F32_1:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            input2.reload(UnaryTestsF32::INPUTS2_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFADD1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F32_ID,mgr);
         break;

         case TEST_MAT_SUB_F32_2:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            input2.reload(UnaryTestsF32::INPUTS2_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFSUB1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F32_ID,mgr);
         break;

         case TEST_MAT_SCALE_F32_3:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFSCALE1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
         break;

         case TEST_MAT_TRANS_F32_4:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFTRANS1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
         break;

         case TEST_MAT_INVERSE_F32_5:
            input1.reload(UnaryTestsF32::INPUTSINV_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSINVERT1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFINV1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
         break;

         case TEST_MAT_VEC_MULT_F32_6:
            input1.reload(UnaryTestsF32::INPUTS1_F32_ID,mgr);
            input2.reload(UnaryTestsF32::INPUTVEC1_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFVECMUL1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM,UnaryTestsF32::TMPB_F32_ID,mgr);
         break;

          case TEST_MAT_CMPLX_TRANS_F32_7:
            input1.reload(UnaryTestsF32::INPUTSC1_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSUNARY1_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFTRANSC1_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
         break;

         case TEST_MAT_CHOLESKY_DPO_F32_8:
            input1.reload(UnaryTestsF32::INPUTSCHOLESKY1_DPO_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSCHOLESKY1_DPO_S16_ID,mgr);

            ref.reload(UnaryTestsF32::REFCHOLESKY1_DPO_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);

            
         break;

         case TEST_SOLVE_UPPER_TRIANGULAR_F32_9:
            input1.reload(UnaryTestsF32::INPUT_UT_DPO_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSCHOLESKY1_DPO_S16_ID,mgr);
            input2.reload(UnaryTestsF32::INPUT_RNDA_DPO_F32_ID,mgr);

            ref.reload(UnaryTestsF32::REF_UTINV_DPO_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F32_ID,mgr);
         break;

         case TEST_SOLVE_LOWER_TRIANGULAR_F32_10:
            input1.reload(UnaryTestsF32::INPUT_LT_DPO_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSCHOLESKY1_DPO_S16_ID,mgr);
            input2.reload(UnaryTestsF32::INPUT_RNDA_DPO_F32_ID,mgr);

            ref.reload(UnaryTestsF32::REF_LTINV_DPO_F32_ID,mgr);

            output.create(ref.nbSamples(),UnaryTestsF32::OUT_F32_ID,mgr);
            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            b.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F32_ID,mgr);
         break;

         case TEST_MAT_LDL_F32_11:
            // Definite positive test
            input1.reload(UnaryTestsF32::INPUTSCHOLESKY1_DPO_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSCHOLESKY1_DPO_S16_ID,mgr);

            outputll.create(input1.nbSamples(),UnaryTestsF32::LL_F32_ID,mgr);
            outputd.create(input1.nbSamples(),UnaryTestsF32::D_F32_ID,mgr);
            outputp.create(input1.nbSamples(),UnaryTestsF32::PERM_S16_ID,mgr);

            outputa.create(input1.nbSamples(),UnaryTestsF32::OUTA_F64_ID,mgr);
            outputb.create(input1.nbSamples(),UnaryTestsF32::OUTB_F64_ID,mgr);

            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            
            tmpapat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F64_ID,mgr);
            tmpbpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPC_F64_ID,mgr);
            tmpcpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPD_F64_ID,mgr);

            this->snrRel=REL_ERROR_LDLT;
            this->snrAbs=ABS_ERROR_LDLT;

         break;

         case TEST_MAT_LDL_F32_12:
            // Semi definite positive test
            input1.reload(UnaryTestsF32::INPUTSCHOLESKY1_SDPO_F32_ID,mgr);
            dims.reload(UnaryTestsF32::DIMSCHOLESKY1_SDPO_S16_ID,mgr);
           
            outputll.create(input1.nbSamples(),UnaryTestsF32::LL_F32_ID,mgr);
            outputd.create(input1.nbSamples(),UnaryTestsF32::D_F32_ID,mgr);
            outputp.create(input1.nbSamples(),UnaryTestsF32::PERM_S16_ID,mgr);

            outputa.create(input1.nbSamples(),UnaryTestsF32::OUTA_F64_ID,mgr);
            outputb.create(input1.nbSamples(),UnaryTestsF32::OUTB_F64_ID,mgr);

            a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPA_F32_ID,mgr);
            
            tmpapat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPB_F64_ID,mgr);
            tmpbpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPC_F64_ID,mgr);
            tmpcpat.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryTestsF32::TMPD_F64_ID,mgr);

            this->snrRel=REL_ERROR_LDLT_SPDO;
            this->snrAbs=ABS_ERROR_LDLT_SDPO;


         break;

      }
       

    
    }

    void UnaryTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       (void)mgr;

       switch(id)
       {
        case TEST_MAT_LDL_F32_11:
          //outputll.dump(mgr);
        break;
       }
       //output.dump(mgr);
    }
