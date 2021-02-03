#include "BinaryF16.h"
#include "Error.h"

   
    void BinaryF16::test_mat_mult_f16()
    {     
      arm_mat_mult_f16(&this->in1,&this->in2,&this->out);
    } 

  
    void BinaryF16::test_mat_cmplx_mult_f16()
    {     
      arm_mat_cmplx_mult_f16(&this->in1,&this->in2,&this->out);
    } 

    
    void BinaryF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbi = *it++;
       this->nbc = *it;

       switch(id)
       {
          case BinaryF16::TEST_MAT_CMPLX_MULT_F16_2:
            input1.reload(BinaryF16::INPUTAC_F16_ID,mgr,2*this->nbr*this->nbi);
            input2.reload(BinaryF16::INPUTBC_F16_ID,mgr,2*this->nbi*this->nbc);
            output.create(2*this->nbr*this->nbc,BinaryF16::OUT_F16_ID,mgr);
          break;

          default:
            input1.reload(BinaryF16::INPUTA_F16_ID,mgr,this->nbr*this->nbi);
            input2.reload(BinaryF16::INPUTB_F16_ID,mgr,this->nbi*this->nbc);
            output.create(this->nbr*this->nbc,BinaryF16::OUT_F16_ID,mgr);

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

    void BinaryF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       (void)mgr;
    }
