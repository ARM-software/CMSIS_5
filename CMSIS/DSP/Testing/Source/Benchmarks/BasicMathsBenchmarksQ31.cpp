#include "BasicMathsBenchmarksQ31.h"
#include "Error.h"

   
    void BasicMathsBenchmarksQ31::vec_mult_q31()
    {
       arm_mult_q31(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ31::vec_add_q31()
    {
       arm_add_q31(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ31::vec_sub_q31()
    {
       arm_sub_q31(this->inp1,this->inp2,this->outp,this->nb); 
    } 

    void BasicMathsBenchmarksQ31::vec_abs_q31()
    {
       arm_abs_q31(this->inp1,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ31::vec_negate_q31()
    {
       arm_negate_q31(this->inp1,this->outp,this->nb);
    }

    void BasicMathsBenchmarksQ31::vec_offset_q31()
    {
       arm_offset_q31(this->inp1,1.0,this->outp,this->nb); 
    }

   void BasicMathsBenchmarksQ31::vec_scale_q31()
    {
       arm_scale_q31(this->inp1,0x45,1,this->outp,this->nb);
    }

    void BasicMathsBenchmarksQ31::vec_dot_q31()
    {
       q63_t result;

       arm_dot_prod_q31(this->inp1,this->inp2,this->nb,&result);
    }

  
    
    void BasicMathsBenchmarksQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksQ31::INPUT1_Q31_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksQ31::INPUT2_Q31_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksQ31::OUT_SAMPLES_Q31_ID,mgr);

       switch(id)
       {
          case BasicMathsBenchmarksQ31::VEC_MULT_Q31_1:
          case BasicMathsBenchmarksQ31::VEC_ADD_Q31_2:
          case BasicMathsBenchmarksQ31::VEC_SUB_Q31_3:
          case BasicMathsBenchmarksQ31::VEC_ABS_Q31_4:
          case BasicMathsBenchmarksQ31::VEC_OFFSET_Q31_6:
          case BasicMathsBenchmarksQ31::VEC_SCALE_Q31_7:
             this->inp1=input1.ptr();
             this->inp2=input2.ptr();
             this->outp=output.ptr();
          break;

          case BasicMathsBenchmarksQ31::VEC_NEGATE_Q31_5:
             this->inp1=input1.ptr();
             this->outp=output.ptr();
          break;

          case BasicMathsBenchmarksQ31::VEC_DOT_Q31_8:
             this->inp1=input1.ptr();
             this->inp2=input2.ptr();
          break;
       }
       
    }

    void BasicMathsBenchmarksQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
