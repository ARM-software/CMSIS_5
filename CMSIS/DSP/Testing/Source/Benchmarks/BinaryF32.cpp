#include "BinaryF32.h"
#include "Error.h"

   
    void BinaryF32::test_mat_mult_f32()
    {     
      arm_mat_mult_f32(&this->in1,&this->in2,&this->out);
    } 

    void BinaryF32::test_mat_cmplx_mult_f32()
    {     
      arm_mat_cmplx_mult_f32(&this->in1,&this->in2,&this->out);
    } 

    
    void BinaryF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbi = *it++;
       this->nbc = *it;

       switch(id)
       {
          case BinaryF32::TEST_MAT_CMPLX_MULT_F32_2:
            input1.reload(BinaryF32::INPUTAC_F32_ID,mgr,2*this->nbr*this->nbi);
            input2.reload(BinaryF32::INPUTBC_F32_ID,mgr,2*this->nbi*this->nbc);
            output.create(2*this->nbr*this->nbc,BinaryF32::OUT_F32_ID,mgr);
          break;

          default:
            input1.reload(BinaryF32::INPUTA_F32_ID,mgr,this->nbr*this->nbi);
            input2.reload(BinaryF32::INPUTB_F32_ID,mgr,this->nbi*this->nbc);
            output.create(this->nbr*this->nbc,BinaryF32::OUT_F32_ID,mgr);

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

    void BinaryF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
