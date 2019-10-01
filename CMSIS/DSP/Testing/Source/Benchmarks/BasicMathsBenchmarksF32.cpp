#include "BasicMathsBenchmarksF32.h"
#include "Error.h"

   
    void BasicMathsBenchmarksF32::vec_mult_f32()
    {     
       arm_mult_f32(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksF32::vec_add_f32()
    {
       arm_add_f32(inp1,inp2,outp,this->nb);
    } 

    void BasicMathsBenchmarksF32::vec_sub_f32()
    {
       arm_sub_f32(inp1,inp2,outp,this->nb);
    } 

    void BasicMathsBenchmarksF32::vec_abs_f32()
    {
       arm_abs_f32(inp1,outp,this->nb);
    } 

    void BasicMathsBenchmarksF32::vec_negate_f32()
    {
       arm_negate_f32(inp1,outp,this->nb);
    }

    void BasicMathsBenchmarksF32::vec_offset_f32()
    {
       arm_offset_f32(inp1,1.0,outp,this->nb);
    }

    void BasicMathsBenchmarksF32::vec_scale_f32()
    {
       arm_scale_f32(inp1,1.0,outp,this->nb);        
    }

    void BasicMathsBenchmarksF32::vec_dot_f32()
    {
       float32_t result;

       arm_dot_prod_f32(inp1,inp2,this->nb,&result);   

    }

    
    void BasicMathsBenchmarksF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksF32::INPUT2_F32_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

       switch(id)
       {
         case BasicMathsBenchmarksF32::VEC_MULT_F32_1:
         case BasicMathsBenchmarksF32::VEC_ADD_F32_2:
         case BasicMathsBenchmarksF32::VEC_SUB_F32_3:
         case BasicMathsBenchmarksF32::VEC_ABS_F32_4:
         case BasicMathsBenchmarksF32::VEC_OFFSET_F32_6:
         case BasicMathsBenchmarksF32::VEC_SCALE_F32_7:

           /* This an overhead doing this because ptr() function is doing lot of checks
            to ensure patterns are fresh.
            So for small benchmark lengths it is better doing it in the setUp function
           */
           this->inp1=input1.ptr();
           this->inp2=input2.ptr();
           this->outp=output.ptr();
         break;
        
         case BasicMathsBenchmarksF32::VEC_NEGATE_F32_5:
           this->inp1=input1.ptr();
           this->outp=output.ptr();
         break;

         case BasicMathsBenchmarksF32::VEC_DOT_F32_8:
           this->inp1=input1.ptr();
           this->inp2=input2.ptr();
         break;
       }
       
    }

    void BasicMathsBenchmarksF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
