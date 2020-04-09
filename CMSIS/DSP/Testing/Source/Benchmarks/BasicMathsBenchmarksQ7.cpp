#include "BasicMathsBenchmarksQ7.h"
#include "Error.h"

   
    void BasicMathsBenchmarksQ7::vec_mult_q7()
    {
       arm_mult_q7(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ7::vec_add_q7()
    {
       arm_add_q7(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ7::vec_sub_q7()
    {
       arm_sub_q7(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ7::vec_abs_q7()
    {
       arm_abs_q7(this->inp1,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ7::vec_negate_q7()
    {
       arm_negate_q7(this->inp1,this->outp,this->nb);
    }

    void BasicMathsBenchmarksQ7::vec_offset_q7()
    {
       arm_offset_q7(this->inp1,1.0,this->outp,this->nb);
    }

   void BasicMathsBenchmarksQ7::vec_scale_q7()
    {
       arm_scale_q7(this->inp1,0x45,1,this->outp,this->nb);
    }

    void BasicMathsBenchmarksQ7::vec_dot_q7()
    {
       
       q31_t result;

       arm_dot_prod_q7(this->inp1,this->inp2,this->nb,&result);
        
    }

  
    
    void BasicMathsBenchmarksQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksQ7::INPUT1_Q7_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksQ7::INPUT2_Q7_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksQ7::OUT_SAMPLES_Q7_ID,mgr);

       switch(id)
       {
          case BasicMathsBenchmarksQ7::VEC_MULT_Q7_1:
          case BasicMathsBenchmarksQ7::VEC_ADD_Q7_2:
          case BasicMathsBenchmarksQ7::VEC_SUB_Q7_3:
          case BasicMathsBenchmarksQ7::VEC_ABS_Q7_4:
          case BasicMathsBenchmarksQ7::VEC_OFFSET_Q7_6:
          case BasicMathsBenchmarksQ7::VEC_SCALE_Q7_7:
             this->inp1=input1.ptr();
             this->inp2=input2.ptr();
             this->outp=output.ptr();
          break;
          case BasicMathsBenchmarksQ7::VEC_NEGATE_Q7_5:
             this->inp1=input1.ptr();
             this->outp=output.ptr();
          break;
          case BasicMathsBenchmarksQ7::VEC_DOT_Q7_8:
             this->inp1=input1.ptr();
             this->inp2=input2.ptr();
          break;
       }
       
    }

    void BasicMathsBenchmarksQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
