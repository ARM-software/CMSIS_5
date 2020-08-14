#include "UnaryF16.h"
#include "Error.h"

   
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
          break;
          case TEST_MAT_TRANS_F16_3:
              input1.reload(UnaryF16::INPUTA_F16_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryF16::OUT_F16_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 
          break;
          case TEST_MAT_CMPLX_TRANS_F16_7:
              input1.reload(UnaryF16::INPUTAC_F16_ID,mgr,2*this->nbr*this->nbc);
              output.create(2*this->nbr*this->nbc,UnaryF16::OUT_F16_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 
          break;
          default:
              input1.reload(UnaryF16::INPUTA_F16_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryF16::OUT_F16_ID,mgr);
              
              this->out.numRows = this->nbr;
              this->out.numCols = this->nbc;
              this->out.pData = output.ptr(); 
          break;
       }


       

       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbc;
       this->in1.pData = input1.ptr();   

      
    }

    void UnaryF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
