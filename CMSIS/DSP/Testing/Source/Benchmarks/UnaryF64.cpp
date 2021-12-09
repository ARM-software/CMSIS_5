#include "UnaryF64.h"
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

    void UnaryF64::test_mat_inverse_f64()
    {     
       arm_mat_inverse_f64(&this->in1,&this->out);
    } 

    void UnaryF64::test_mat_cholesky_dpo_f64()
    {
        arm_mat_cholesky_f64(&this->in1,&this->out);
    }

    void UnaryF64::test_solve_upper_triangular_f64()
    {
        arm_mat_solve_upper_triangular_f64(&this->in1,&this->in2,&this->out);
    }

    void UnaryF64::test_solve_lower_triangular_f64()
    {
        arm_mat_solve_lower_triangular_f64(&this->in1,&this->in2,&this->out);
    }

  
    void UnaryF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       switch(id)
       {
          case TEST_MAT_INVERSE_F64_1:

            input1.reload(UnaryF64::INPUTA_F64_ID,mgr,this->nbr*this->nbc);
     
            
            output.create(this->nbr*this->nbc,UnaryF64::OUT_F64_ID,mgr);
     
            this->in1.numRows = this->nbr;
            this->in1.numCols = this->nbc;
            this->in1.pData = input1.ptr();   
     
            this->out.numRows = this->nbr;
            this->out.numCols = this->nbc;
            this->out.pData = output.ptr(); 
          break;

          case TEST_MAT_CHOLESKY_DPO_F64_2:
          {
            int offset=14;
            float64_t *p;
            float64_t *aPtr;
            input1.reload(UnaryF64::INPUTSCHOLESKY1_DPO_F64_ID,mgr);
            output.create(this->nbc * this->nbr,UnaryF64::OUT_F64_ID,mgr);

            a.create(this->nbr*this->nbc,UnaryF64::TMPA_F64_ID,mgr);

            /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
            offset=cholesky_offset(this->nbr);
            
             p = input1.ptr();
             aPtr = a.ptr();



             memcpy(aPtr,p + offset,sizeof(float64_t)*this->nbr*this->nbr);

             this->out.numRows = this->nbr;
             this->out.numCols = this->nbc;
             this->out.pData = output.ptr(); 

             this->in1.numRows = this->nbr;
             this->in1.numCols = this->nbc;
             this->in1.pData = aPtr; 

            

          }
          break;

          case TEST_SOLVE_UPPER_TRIANGULAR_F64_3:
          {
             int offset=14;
             float64_t *p;
             float64_t *aPtr;
             float64_t *bPtr;

             input1.reload(UnaryF64::INPUT_UT_DPO_F64_ID,mgr);
             input2.reload(UnaryF64::INPUT_RNDA_DPO_F64_ID,mgr);
             output.create(this->nbc * this->nbr,UnaryF64::OUT_F64_ID,mgr);
 
             a.create(this->nbr*this->nbc,UnaryF64::TMPA_F64_ID,mgr);
             b.create(this->nbr*this->nbc,UnaryF64::TMPB_F64_ID,mgr);

             /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
             offset=cholesky_offset(this->nbr);

             p = input1.ptr();
             aPtr = a.ptr();
             memcpy(aPtr,&p[offset],sizeof(float64_t)*this->nbr*this->nbr);

             p = input2.ptr();
             bPtr = b.ptr();
             memcpy(bPtr,&p[offset],sizeof(float64_t)*this->nbr*this->nbr);

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

          case TEST_SOLVE_LOWER_TRIANGULAR_F64_4:
          {
             int offset=14;
             float64_t *p;
             float64_t *aPtr;
             float64_t *bPtr;

             input1.reload(UnaryF64::INPUT_LT_DPO_F64_ID,mgr);
             input2.reload(UnaryF64::INPUT_RNDA_DPO_F64_ID,mgr);
             output.create(this->nbc * this->nbr,UnaryF64::OUT_F64_ID,mgr);
 
             a.create(this->nbr*this->nbc,UnaryF64::TMPA_F64_ID,mgr);
             b.create(this->nbr*this->nbc,UnaryF64::TMPB_F64_ID,mgr);

             /* Offsets must be coherent with the sizes used in python script
                Matrix.py for pattern generation */
             offset=cholesky_offset(this->nbr);
            
             p = input1.ptr();
             aPtr = a.ptr();
             memcpy(aPtr,&p[offset],sizeof(float64_t)*this->nbr*this->nbr);

             p = input2.ptr();
             bPtr = b.ptr();
             memcpy(bPtr,&p[offset],sizeof(float64_t)*this->nbr*this->nbr);

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
        }
    }

    void UnaryF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
