#include "BinaryQ15.h"
#include "Error.h"

   
    void BinaryQ15::test_mat_mult_q15()
    {     
      arm_mat_mult_q15(&this->in1,&this->in2,&this->out,this->pState);
    } 

    void BinaryQ15::test_mat_cmplx_mult_q15()
    {     
      arm_mat_cmplx_mult_q15(&this->in1,&this->in2,&this->out,this->pState);
    } 

    void BinaryQ15::test_mat_mult_fast_q15()
    {     
      arm_mat_mult_fast_q15(&this->in1,&this->in2,&this->out,this->pState);
    }

    
    void BinaryQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbi = *it++;
       this->nbc = *it;

       switch(id)
       {
          case BinaryQ15::TEST_MAT_CMPLX_MULT_Q15_2:
            input1.reload(BinaryQ15::INPUTAC_Q15_ID,mgr,2*this->nbr*this->nbi);
            input2.reload(BinaryQ15::INPUTBC_Q15_ID,mgr,2*this->nbi*this->nbc);
            output.create(2*this->nbr*this->nbc,BinaryQ15::OUT_Q15_ID,mgr);
            state.create(2*this->nbi*this->nbc,BinaryQ15::OUT_Q15_ID,mgr);
          break;

          default:
            input1.reload(BinaryQ15::INPUTA_Q15_ID,mgr,this->nbr*this->nbi);
            input2.reload(BinaryQ15::INPUTB_Q15_ID,mgr,this->nbi*this->nbc);
            state.create(this->nbi*this->nbc,BinaryQ15::OUT_Q15_ID,mgr);
            output.create(this->nbr*this->nbc,BinaryQ15::OUT_Q15_ID,mgr);

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

       this->pState = state.ptr();
    }

    void BinaryQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
