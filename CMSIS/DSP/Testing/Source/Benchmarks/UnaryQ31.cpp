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

    void UnaryQ31::test_mat_add_q31()
    {     
       arm_mat_add_q31(&this->in1,&this->in1,&this->out);
    } 

    void UnaryQ31::test_mat_sub_q31()
    {     
       arm_mat_sub_q31(&this->in1,&this->in1,&this->out);
    } 
    
    void UnaryQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       input1.reload(UnaryQ31::INPUTA_Q31_ID,mgr,this->nbr*this->nbc);

       
       output.create(this->nbr*this->nbc,UnaryQ31::OUT_Q31_ID,mgr);

       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbc;
       this->in1.pData = input1.ptr();   

       this->out.numRows = this->nbr;
       this->out.numCols = this->nbc;
       this->out.pData = output.ptr(); 
    }

    void UnaryQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
