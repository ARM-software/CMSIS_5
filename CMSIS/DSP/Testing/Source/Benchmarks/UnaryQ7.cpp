#include "UnaryQ7.h"
#include "Error.h"

   
    /*void UnaryQ7::test_mat_scale_q7()
    {     
       arm_mat_scale_q7(&this->in1,0x4000,1,&this->out);
    } 
*/
    void UnaryQ7::test_mat_trans_q7()
    {     
       arm_mat_trans_q7(&this->in1,&this->out);
    } 
/*
    void UnaryQ7::test_mat_add_q7()
    {     
       arm_mat_add_q7(&this->in1,&this->in1,&this->out);
    } 

    void UnaryQ7::test_mat_sub_q7()
    {     
       arm_mat_sub_q7(&this->in1,&this->in1,&this->out);
    } 
    */

    void UnaryQ7::test_mat_vec_mult_q7()
    {     
       arm_mat_vec_mult_q7(&this->in1, vecp, outp);
    }
    

    void UnaryQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbr = *it++;
       this->nbc = *it;

       switch(id)
       {
          case TEST_MAT_VEC_MULT_Q7_2:
             vec.reload(UnaryQ7::INPUTVEC1_Q7_ID,mgr,this->nbc);
             output.create(this->nbr,UnaryQ7::OUT_Q7_ID,mgr);
             vecp=vec.ptr();
             outp=output.ptr();
          break;
          default:
              output.create(this->nbr*this->nbc,UnaryQ7::OUT_Q7_ID,mgr);
              
              this->out.numRows = this->nbr;
              this->out.numCols = this->nbc;
              this->out.pData = output.ptr(); 
          break;
       }

       input1.reload(UnaryQ7::INPUTA_Q7_ID,mgr,this->nbr*this->nbc);

      
       this->in1.numRows = this->nbr;
       this->in1.numCols = this->nbc;
       this->in1.pData = input1.ptr();   

    }

    void UnaryQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
