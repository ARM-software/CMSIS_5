#include "UnaryQ15.h"
#include "Error.h"

   
    void UnaryQ15::test_mat_scale_q15()
    {     
       arm_mat_scale_q15(&this->in1,0x4000,1,&this->out);
    } 

    void UnaryQ15::test_mat_trans_q15()
    {     
       arm_mat_trans_q15(&this->in1,&this->out);
    } 

    void UnaryQ15::test_mat_cmplx_trans_q15()
    {     
       arm_mat_cmplx_trans_q15(&this->in1,&this->out);
    } 

    void UnaryQ15::test_mat_add_q15()
    {     
       arm_mat_add_q15(&this->in1,&this->in1,&this->out);
    } 

    void UnaryQ15::test_mat_sub_q15()
    {     
       arm_mat_sub_q15(&this->in1,&this->in1,&this->out);
    } 

    void UnaryQ15::test_mat_vec_mult_q15()
    {     
       arm_mat_vec_mult_q15(&this->in1, vecp, outp);
    }
    
    void UnaryQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       switch(id)
       {
          case TEST_MAT_VEC_MULT_Q15_5:
             input1.reload(UnaryQ15::INPUTA_Q15_ID,mgr,this->nbr*this->nbc);
             vec.reload(UnaryQ15::INPUTVEC1_Q15_ID,mgr,this->nbc);
             output.create(this->nbr,UnaryQ15::OUT_Q15_ID,mgr);
             vecp=vec.ptr();
             outp=output.ptr();
          break;
          case TEST_MAT_TRANS_Q15_2:
              input1.reload(UnaryQ15::INPUTA_Q15_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryQ15::OUT_Q15_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 
          break;
          case TEST_MAT_CMPLX_TRANS_Q15_6:
              input1.reload(UnaryQ15::INPUTAC_Q15_ID,mgr,2*this->nbr*this->nbc);
              output.create(2*this->nbr*this->nbc,UnaryQ15::OUT_Q15_ID,mgr);
              
              this->out.numRows = this->nbc;
              this->out.numCols = this->nbr;
              this->out.pData = output.ptr(); 
          break;
          default:
              input1.reload(UnaryQ15::INPUTA_Q15_ID,mgr,this->nbr*this->nbc);
              output.create(this->nbr*this->nbc,UnaryQ15::OUT_Q15_ID,mgr);
              
              this->out.numRows = this->nbr;
              this->out.numCols = this->nbc;
              this->out.pData = output.ptr(); 
          break;
       }

       

       
       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbc;
       this->in1.pData = input1.ptr();   

      
    }

    void UnaryQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
