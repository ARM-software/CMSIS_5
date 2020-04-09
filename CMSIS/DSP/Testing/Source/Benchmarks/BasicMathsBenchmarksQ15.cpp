#include "BasicMathsBenchmarksQ15.h"
#include "Error.h"

   
    void BasicMathsBenchmarksQ15::vec_mult_q15()
    {
       arm_mult_q15(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ15::vec_add_q15()
    {
       arm_add_q15(this->inp1,this->inp2,this->outp,this->nb); 
    } 

    void BasicMathsBenchmarksQ15::vec_sub_q15()
    {
       arm_sub_q15(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ15::vec_abs_q15()
    {
       arm_abs_q15(this->inp1,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksQ15::vec_negate_q15()
    {
       arm_negate_q15(this->inp1,this->outp,this->nb);
    }

    void BasicMathsBenchmarksQ15::vec_offset_q15()
    {
       arm_offset_q15(this->inp1,1.0,this->outp,this->nb);
    }

   void BasicMathsBenchmarksQ15::vec_scale_q15()
    {
       arm_scale_q15(this->inp1,0x45,1,this->outp,this->nb); 
    }

    void BasicMathsBenchmarksQ15::vec_dot_q15()
    {
       q63_t result;

       arm_dot_prod_q15(this->inp1,this->inp2,this->nb,&result);
    }

  
    
    void BasicMathsBenchmarksQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksQ15::INPUT1_Q15_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksQ15::INPUT2_Q15_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksQ15::OUT_SAMPLES_Q15_ID,mgr);

       switch(id) {
         case BasicMathsBenchmarksQ15::VEC_MULT_Q15_1:
         case BasicMathsBenchmarksQ15::VEC_ADD_Q15_2:
         case BasicMathsBenchmarksQ15::VEC_SUB_Q15_3:
         case BasicMathsBenchmarksQ15::VEC_ABS_Q15_4:
         case BasicMathsBenchmarksQ15::VEC_SCALE_Q15_7:
         case BasicMathsBenchmarksQ15::VEC_OFFSET_Q15_6:
           this->inp1=input1.ptr();
           this->inp2=input2.ptr();
           this->outp=output.ptr();
         break;
         case BasicMathsBenchmarksQ15::VEC_NEGATE_Q15_5:
           this->inp1=input1.ptr();
           this->outp=output.ptr();
         break;
         case BasicMathsBenchmarksQ15::VEC_DOT_Q15_8:
           this->inp1=input1.ptr();
           this->inp2=input2.ptr();
         break;
       }
       
    }

    void BasicMathsBenchmarksQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
