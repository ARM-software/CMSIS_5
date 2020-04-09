#include "BinaryQ31.h"
#include "Error.h"

   
    void BinaryQ31::test_mat_mult_q31()
    {     
      arm_mat_mult_q31(&this->in1,&this->in2,&this->out);
    } 

    void BinaryQ31::test_mat_cmplx_mult_q31()
    {     
      arm_mat_cmplx_mult_q31(&this->in1,&this->in2,&this->out);
    } 

    void BinaryQ31::test_mat_mult_fast_q31()
    {     
      arm_mat_mult_fast_q31(&this->in1,&this->in2,&this->out);
    }

    
    void BinaryQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbi = *it++;
       this->nbc = *it;

       switch(id)
       {
          case BinaryQ31::TEST_MAT_CMPLX_MULT_Q31_2:
            input1.reload(BinaryQ31::INPUTAC_Q31_ID,mgr,2*this->nbr*this->nbi);
            input2.reload(BinaryQ31::INPUTBC_Q31_ID,mgr,2*this->nbi*this->nbc);
            output.create(2*this->nbr*this->nbc,BinaryQ31::OUT_Q31_ID,mgr);
          break;

          default:
            input1.reload(BinaryQ31::INPUTA_Q31_ID,mgr,this->nbr*this->nbi);
            input2.reload(BinaryQ31::INPUTB_Q31_ID,mgr,this->nbi*this->nbc);
            output.create(this->nbr*this->nbc,BinaryQ31::OUT_Q31_ID,mgr);

       } 
       

       

       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbi;
       this->in1.pData = input1.ptr();   

       this->in2.numRows = this->nbi;
       this->in2.numCols = this->nbc;
       this->in2.pData = input2.ptr();   

       this->out.numRows = this->nbr;
       this->out.numCols = this->nbc;
       this->out.pData = output.ptr();     
    }

    void BinaryQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
