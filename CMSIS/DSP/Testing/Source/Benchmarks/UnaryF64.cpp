#include "UnaryF64.h"
#include "Error.h"

    void UnaryF64::test_mat_inverse_f64()
    {     
       arm_mat_inverse_f64(&this->in1,&this->out);
    } 

  
    void UnaryF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       input1.reload(UnaryF64::INPUTA_F64_ID,mgr,this->nbr*this->nbc);

       
       output.create(this->nbr*this->nbc,UnaryF64::OUT_F64_ID,mgr);

       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbc;
       this->in1.pData = input1.ptr();   

       this->out.numRows = this->nbr;
       this->out.numCols = this->nbc;
       this->out.pData = output.ptr(); 
    }

    void UnaryF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
