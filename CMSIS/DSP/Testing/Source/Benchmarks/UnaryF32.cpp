#include "UnaryF32.h"
#include "Error.h"

   
    void UnaryF32::test_mat_scale_f32()
    {     
       arm_mat_scale_f32(&this->in1,0.5,&this->out);
    } 

    void UnaryF32::test_mat_inverse_f32()
    {     
       arm_mat_inverse_f32(&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_trans_f32()
    {     
       arm_mat_trans_f32(&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_add_f32()
    {     
       arm_mat_add_f32(&this->in1,&this->in1,&this->out);
    } 

    void UnaryF32::test_mat_sub_f32()
    {     
       arm_mat_sub_f32(&this->in1,&this->in1,&this->out);
    } 
    
    void UnaryF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       input1.reload(UnaryF32::INPUTA_F32_ID,mgr,this->nbr*this->nbc);

       
       output.create(this->nbr*this->nbc,UnaryF32::OUT_F32_ID,mgr);

       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbc;
       this->in1.pData = input1.ptr();   

       this->out.numRows = this->nbr;
       this->out.numCols = this->nbc;
       this->out.pData = output.ptr(); 
    }

    void UnaryF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
