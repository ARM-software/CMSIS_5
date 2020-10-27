#include "BasicMathsBenchmarksF16.h"
#include "Error.h"

   
    void BasicMathsBenchmarksF16::vec_mult_f16()
    {     
       arm_mult_f16(this->inp1,this->inp2,this->outp,this->nb);
    } 

    void BasicMathsBenchmarksF16::vec_add_f16()
    {
       arm_add_f16(inp1,inp2,outp,this->nb);
    } 

    void BasicMathsBenchmarksF16::vec_sub_f16()
    {
       arm_sub_f16(inp1,inp2,outp,this->nb);
    } 

    void BasicMathsBenchmarksF16::vec_abs_f16()
    {
       arm_abs_f16(inp1,outp,this->nb);
    } 

    void BasicMathsBenchmarksF16::vec_negate_f16()
    {
       arm_negate_f16(inp1,outp,this->nb);
    }

    void BasicMathsBenchmarksF16::vec_offset_f16()
    {
       arm_offset_f16(inp1,1.0,outp,this->nb);
    }

    void BasicMathsBenchmarksF16::vec_scale_f16()
    {
       arm_scale_f16(inp1,1.0,outp,this->nb);        
    }

    void BasicMathsBenchmarksF16::vec_dot_f16()
    {
       float16_t result;

       arm_dot_prod_f16(inp1,inp2,this->nb,&result);   

    }

    
    void BasicMathsBenchmarksF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {

       this->setForceInCache(true);
       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksF16::INPUT1_F16_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksF16::INPUT2_F16_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksF16::OUT_SAMPLES_F16_ID,mgr);

       switch(id)
       {
         case BasicMathsBenchmarksF16::VEC_MULT_F16_1:
         case BasicMathsBenchmarksF16::VEC_ADD_F16_2:
         case BasicMathsBenchmarksF16::VEC_SUB_F16_3:
         case BasicMathsBenchmarksF16::VEC_ABS_F16_4:
         case BasicMathsBenchmarksF16::VEC_OFFSET_F16_6:
         case BasicMathsBenchmarksF16::VEC_SCALE_F16_7:

           /* This an overhead doing this because ptr() function is doing lot of checks
            to ensure patterns are fresh.
            So for small benchmark lengths it is better doing it in the setUp function
           */
           this->inp1=input1.ptr();
           this->inp2=input2.ptr();
           this->outp=output.ptr();

         break;
        
         case BasicMathsBenchmarksF16::VEC_NEGATE_F16_5:
           this->inp1=input1.ptr();
           this->outp=output.ptr();
         break;

         case BasicMathsBenchmarksF16::VEC_DOT_F16_8:
           this->inp1=input1.ptr();
           this->inp2=input2.ptr();
         break;
       }
       
    }

    void BasicMathsBenchmarksF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
    }
