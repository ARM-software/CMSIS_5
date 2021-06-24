#include "UnaryF16.h"
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

    void UnaryF16::test_mat_scale_f16()
    {     
       arm_mat_scale_f16(&this->in1,0.5,&this->out);
    } 

    void UnaryF16::test_mat_inverse_f16()
    {     
       arm_mat_inverse_f16(&this->in1,&this->out);
    } 

    void UnaryF16::test_mat_trans_f16()
    {     
       arm_mat_trans_f16(&this->in1,&this->out);
    } 

    void UnaryF16::test_mat_cmplx_trans_f16()
    {     
       arm_mat_cmplx_trans_f16(&this->in1,&this->out);
    } 

    void UnaryF16::test_mat_add_f16()
    {     
       arm_mat_add_f16(&this->in1,&this->in1,&this->out);
    } 

    void UnaryF16::test_mat_sub_f16()
    {     
       arm_mat_sub_f16(&this->in1,&this->in1,&this->out);
    } 

    void UnaryF16::test_mat_vec_mult_f16()
    {     
       arm_mat_vec_mult_f16(&this->in1, vecp, outp);
    } 

    void UnaryF16::test_mat_cholesky_dpo_f16()
    {
        arm_mat_cholesky_f16(&this->in1,&this->out);
    }

    void UnaryF16::test_solve_upper_triangular_f16()
    {
        arm_mat_solve_upper_triangular_f16(&this->in1,&this->in2,&this->out);
    }

    void UnaryF16::test_solve_lower_triangular_f16()
    {
        arm_mat_solve_lower_triangular_f16(&this->in1,&this->in2,&this->out);
    }
    
    void UnaryF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       switch(id)
       {
          case TEST_MAT_VEC_MULT_F16_6:
             input1.reload(UnaryF16::INPUTA_F16_ID,mgr,this->nbr*this->nbc);
             vec.reload(UnaryF16::INPUTVEC1_F16_ID,mgr,this->nbc);
             output.create(this->nbr,UnaryF16::OUT_F16_ID,mgr);
             vecp=vec.ptr();
             outp=output.ptr();

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = input1.ptr();   
          break;
          case TEST_MAT_TRANS_F16_3:
              input1.reload(UnaryF16::INPUTA_F16_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryF16::OUT_F16_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 

              this->in1.numRows = this->nbr;
              this->in1.numCols = this->nbc;
              this->in1.pData = input1.ptr();   
          break;
          case TEST_MAT_CMPLX_TRANS_F16_7:
              input1.reload(UnaryF16::INPUTAC_F16_ID,mgr,2*this->nbr*this->nbc);
              output.create(2*this->nbr*this->nbc,UnaryF16::OUT_F16_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 

              this->in1.numRows = this->nbr;
              this->in1.numCols = this->nbc;
              this->in1.pData = input1.ptr();   
          break;

          case TEST_MAT_CHOLESKY_DPO_F16_8:
          {
            int offset=14;
            float16_t *p;
            float16_t *aPtr;
            input1.reload(UnaryF16::INPUTSCHOLESKY1_DPO_F16_ID,mgr);
            output.create(this->nbc * this->nbr,UnaryF16::OUT_F16_ID,mgr);

            a.create(this->nbr*this->nbc,UnaryF16::TMPA_F16_ID,mgr);

            /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
            offset=cholesky_offset(this->nbr);
            
             p = input1.ptr();
             aPtr = a.ptr();



             memcpy(aPtr,p + offset,sizeof(float16_t)*this->nbr*this->nbr);

             this->out.numRows = this->nbr;
             this->out.numCols = this->nbc;
             this->out.pData = output.ptr(); 

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = aPtr; 

            

          }
          break;

          case TEST_SOLVE_UPPER_TRIANGULAR_F16_9:
          {
             int offset=14;
             float16_t *p;
             float16_t *aPtr;
             float16_t *bPtr;

             input1.reload(UnaryF16::INPUT_UT_DPO_F16_ID,mgr);
             input2.reload(UnaryF16::INPUT_RNDA_DPO_F16_ID,mgr);
             output.create(this->nbc * this->nbr,UnaryF16::OUT_F16_ID,mgr);
 
             a.create(this->nbr*this->nbc,UnaryF16::TMPA_F16_ID,mgr);
             b.create(this->nbr*this->nbc,UnaryF16::TMPB_F16_ID,mgr);

             /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
             offset=cholesky_offset(this->nbr);

             p = input1.ptr();
             aPtr = a.ptr();
             memcpy(aPtr,&p[offset],sizeof(float16_t)*this->nbr*this->nbr);

             p = input2.ptr();
             bPtr = b.ptr();
             memcpy(bPtr,&p[offset],sizeof(float16_t)*this->nbr*this->nbr);

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

          case TEST_SOLVE_LOWER_TRIANGULAR_F16_10:
          {
             int offset=14;
             float16_t *p;
             float16_t *aPtr;
             float16_t *bPtr;

             input1.reload(UnaryF16::INPUT_LT_DPO_F16_ID,mgr);
             input2.reload(UnaryF16::INPUT_RNDA_DPO_F16_ID,mgr);
             output.create(this->nbc * this->nbr,UnaryF16::OUT_F16_ID,mgr);
 
             a.create(this->nbr*this->nbc,UnaryF16::TMPA_F16_ID,mgr);
             b.create(this->nbr*this->nbc,UnaryF16::TMPB_F16_ID,mgr);

             /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
             offset=cholesky_offset(this->nbr);
            
             p = input1.ptr();
             aPtr = a.ptr();
             memcpy(aPtr,&p[offset],sizeof(float16_t)*this->nbr*this->nbr);

             p = input2.ptr();
             bPtr = b.ptr();
             memcpy(bPtr,&p[offset],sizeof(float16_t)*this->nbr*this->nbr);

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
          
          default:
              input1.reload(UnaryF16::INPUTA_F16_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryF16::OUT_F16_ID,mgr);
              
              this->out.numRows = this->nbr;
              this->out.numCols = this->nbc;
              this->out.pData = output.ptr(); 

              this->in1.numRows = this->nbr;
              this->in1.numCols = this->nbc;
              this->in1.pData = input1.ptr();   
          break;
       }


       

       

      
    }

    void UnaryF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
    }
