#include "UnaryQ31.h"
#include "Error.h"

   
    void UnaryQ31::test_mat_scale_q31()
    {     
       arm_mat_scale_q31(&this->in1,0x40000000,1,&this->out);
    } 

    void UnaryQ31::test_mat_trans_q31()
    {     
       arm_mat_trans_q31(&this->in1,&this->out);
    } 

    void UnaryQ31::test_mat_cmplx_trans_q31()
    {     
       arm_mat_cmplx_trans_q31(&this->in1,&this->out);
    } 

    void UnaryQ31::test_mat_add_q31()
    {     
       arm_mat_add_q31(&this->in1,&this->in1,&this->out);
    } 

    void UnaryQ31::test_mat_sub_q31()
    {     
       arm_mat_sub_q31(&this->in1,&this->in1,&this->out);
    } 
    
    void UnaryQ31::test_mat_vec_mult_q31()
    {     
       arm_mat_vec_mult_q31(&this->in1, vecp, outp);
    } 

    void UnaryQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       switch(id)
       {
          case TEST_MAT_VEC_MULT_Q31_5:
             input1.reload(UnaryQ31::INPUTA_Q31_ID,mgr,this->nbr*this->nbc);
             vec.reload(UnaryQ31::INPUTVEC1_Q31_ID,mgr,this->nbc);
             output.create(this->nbr,UnaryQ31::OUT_Q31_ID,mgr);
             vecp=vec.ptr();
             outp=output.ptr();
          break;
          case TEST_MAT_TRANS_Q31_2:
              input1.reload(UnaryQ31::INPUTA_Q31_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryQ31::OUT_Q31_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 
          break;
          case TEST_MAT_CMPLX_TRANS_Q31_6:
              input1.reload(UnaryQ31::INPUTAC_Q31_ID,mgr,2*this->nbr*this->nbc);
              output.create(2*this->nbr*this->nbc,UnaryQ31::OUT_Q31_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 
          break;
          default:
              input1.reload(UnaryQ31::INPUTA_Q31_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryQ31::OUT_Q31_ID,mgr);
              
              this->out.numRows = this->nbr;
              this->out.numCols = this->nbc;
              this->out.pData = output.ptr(); 
          break;
       }


      
       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbc;
       this->in1.pData = input1.ptr();   

    }

    void UnaryQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
