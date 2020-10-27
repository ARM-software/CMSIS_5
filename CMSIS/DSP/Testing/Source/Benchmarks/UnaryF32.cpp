#include "UnaryF32.h"
#include "Error.h"

/* Upper bound of maximum matrix dimension used by Python */
#define MAXMATRIXDIM 40

/*

Offset in input test pattern for matrix of dimension d * d.
Must be coherent with Python script Matrix.py

*/
static int cholesky_offset(int d)
{
  int offset=14;
  switch (d)
  {
   case 4:
     offset = 14;
   break;
   case 8:
     offset = 79;
   break;
   case 9:
     offset = 143;
   break;
   case 15:
     offset = 224;
   break;
   case 16:
     offset = 449;
   break;
   default:
    offset = 14;
   break;
  }

  return(offset);
}


    void UnaryF32::test_mat_scale_f32()
    {     
       arm_mat_scale_f32(&this->in1,0.5,&this->out);
    } 

    void UnaryF32::test_mat_inverse_f32()
    {     
       arm_mat_inverse_f32(&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_trans_f32()
    {     
       arm_mat_trans_f32(&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_cmplx_trans_f32()
    {     
       arm_mat_cmplx_trans_f32(&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_add_f32()
    {     
       arm_mat_add_f32(&this->in1,&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_sub_f32()
    {     
       arm_mat_sub_f32(&this->in1,&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_vec_mult_f32()
    {     
       arm_mat_vec_mult_f32(&this->in1, vecp, outp);
    } 

    void UnaryF32::test_mat_cholesky_dpo_f32()
    {
        arm_mat_cholesky_f32(&this->in1,&this->out);
    }

    void UnaryF32::test_solve_upper_triangular_f32()
    {
        arm_mat_solve_upper_triangular_f32(&this->in1,&this->in2,&this->out);
    }

    void UnaryF32::test_solve_lower_triangular_f32()
    {
        arm_mat_solve_lower_triangular_f32(&this->in1,&this->in2,&this->out);
    }

    void UnaryF32::test_ldlt_decomposition_f32()
    {
        arm_mat_ldlt_f32(&this->in1,&this->outll,&this->outd,(uint16_t*)outp);
    }

    
    void UnaryF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();

       this->nbr = *it++;
       this->nbc = *it;
       

       switch(id)
       {
          case TEST_MAT_VEC_MULT_F32_6:
             input1.reload(UnaryF32::INPUTA_F32_ID,mgr,this->nbr*this->nbc);
             vec.reload(UnaryF32::INPUTVEC1_F32_ID,mgr,this->nbc);
             output.create(this->nbr,UnaryF32::OUT_F32_ID,mgr);
             vecp=vec.ptr();
             outp=output.ptr();

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = input1.ptr();   
          break;
          case TEST_MAT_TRANS_F32_3:
              input1.reload(UnaryF32::INPUTA_F32_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryF32::OUT_F32_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 

              this->in1.numRows = this->nbr;
              this->in1.numCols = this->nbc;
              this->in1.pData = input1.ptr();   
          break;
          case TEST_MAT_CMPLX_TRANS_F32_7:
              input1.reload(UnaryF32::INPUTAC_F32_ID,mgr,2*this->nbr*this->nbc);
              output.create(2*this->nbr*this->nbc,UnaryF32::OUT_F32_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 

              this->in1.numRows = this->nbr;
              this->in1.numCols = this->nbc;
              this->in1.pData = input1.ptr();   
          break;

          case TEST_MAT_CHOLESKY_DPO_F32_8:
          {
            int offset=14;
            float32_t *p;
            float32_t *aPtr;
            input1.reload(UnaryF32::INPUTSCHOLESKY1_DPO_F32_ID,mgr);
            output.create(this->nbc * this->nbr,UnaryF32::OUT_F32_ID,mgr);

            a.create(this->nbr*this->nbc,UnaryF32::TMPA_F32_ID,mgr);

            /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
            offset=cholesky_offset(this->nbr);
            
             p = input1.ptr();
             aPtr = a.ptr();



             memcpy(aPtr,p + offset,sizeof(float32_t)*this->nbr*this->nbr);

             this->out.numRows = this->nbr;
             this->out.numCols = this->nbc;
             this->out.pData = output.ptr(); 

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = aPtr; 

            

          }
          break;

          case TEST_SOLVE_UPPER_TRIANGULAR_F32_9:
          {
             int offset=14;
             float32_t *p;
             float32_t *aPtr;
             float32_t *bPtr;

             input1.reload(UnaryF32::INPUT_UT_DPO_F32_ID,mgr);
             input2.reload(UnaryF32::INPUT_RNDA_DPO_F32_ID,mgr);
             output.create(this->nbc * this->nbr,UnaryF32::OUT_F32_ID,mgr);
 
             a.create(this->nbr*this->nbc,UnaryF32::TMPA_F32_ID,mgr);
             b.create(this->nbr*this->nbc,UnaryF32::TMPB_F32_ID,mgr);

             /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
             offset=cholesky_offset(this->nbr);

             p = input1.ptr();
             aPtr = a.ptr();
             memcpy(aPtr,&p[offset],sizeof(float32_t)*this->nbr*this->nbr);

             p = input2.ptr();
             bPtr = b.ptr();
             memcpy(bPtr,&p[offset],sizeof(float32_t)*this->nbr*this->nbr);

             this->out.numRows = this->nbr;
             this->out.numCols = this->nbc;
             this->out.pData = output.ptr(); 

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = aPtr; 

             this->in2.numRows = this->nbr;
             this->in2.numCols = this->nbc;
             this->in2.pData = bPtr; 
          }
          break;

          case TEST_SOLVE_LOWER_TRIANGULAR_F32_10:
          {
             int offset=14;
             float32_t *p;
             float32_t *aPtr;
             float32_t *bPtr;

             input1.reload(UnaryF32::INPUT_LT_DPO_F32_ID,mgr);
             input2.reload(UnaryF32::INPUT_RNDA_DPO_F32_ID,mgr);
             output.create(this->nbc * this->nbr,UnaryF32::OUT_F32_ID,mgr);
 
             a.create(this->nbr*this->nbc,UnaryF32::TMPA_F32_ID,mgr);
             b.create(this->nbr*this->nbc,UnaryF32::TMPB_F32_ID,mgr);

             /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
             offset=cholesky_offset(this->nbr);
            
             p = input1.ptr();
             aPtr = a.ptr();
             memcpy(aPtr,&p[offset],sizeof(float32_t)*this->nbr*this->nbr);

             p = input2.ptr();
             bPtr = b.ptr();
             memcpy(bPtr,&p[offset],sizeof(float32_t)*this->nbr*this->nbr);

             this->out.numRows = this->nbr;
             this->out.numCols = this->nbc;
             this->out.pData = output.ptr(); 

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = aPtr; 

             this->in2.numRows = this->nbr;
             this->in2.numCols = this->nbc;
             this->in2.pData = bPtr; 
          }
          break;

          case TEST_LDLT_DECOMPOSITION_F32_11:
          {
             float32_t *p, *aPtr;
             
             int offset=14;
             input1.reload(UnaryF32::INPUTSCHOLESKY1_DPO_F32_ID,mgr);

            
             outputll.create(this->nbr*this->nbr,UnaryF32::LL_F32_ID,mgr);
             outputd.create(this->nbr*this->nbr,UnaryF32::D_F32_ID,mgr);
             outputp.create(this->nbr,UnaryF32::PERM_S16_ID,mgr);


             a.create(MAXMATRIXDIM*MAXMATRIXDIM,UnaryF32::TMPA_F32_ID,mgr);
             
             /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
             offset=cholesky_offset(this->nbr);
            
             p = input1.ptr();
             aPtr = a.ptr();
             memcpy(aPtr,&p[offset],sizeof(float32_t)*this->nbr*this->nbr);

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = aPtr; 

             this->outll.numRows = this->nbr;
             this->outll.numCols = this->nbc;
             this->outll.pData = outputll.ptr(); 

             this->outd.numRows = this->nbr;
             this->outd.numCols = this->nbc;
             this->outd.pData = outputd.ptr(); 
             
            
             outpp = outputp.ptr(); 
          }
          break;

          default:
              input1.reload(UnaryF32::INPUTA_F32_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryF32::OUT_F32_ID,mgr);
              
              this->out.numRows = this->nbr;
              this->out.numCols = this->nbc;
              this->out.pData = output.ptr(); 

              this->in1.numRows = this->nbr;
              this->in1.numCols = this->nbc;
              this->in1.pData = input1.ptr();   
          break;
       }


       

      

      
    }

    void UnaryF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
