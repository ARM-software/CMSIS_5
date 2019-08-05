#include "BasicMathsBenchmarksF32.h"
#include "Error.h"

   
    void BasicMathsBenchmarksF32::vec_mult_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *inp2=input2.ptr();
       float32_t *outp=output.ptr();


       arm_mult_f32(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksF32::vec_add_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *inp2=input2.ptr();
       float32_t *outp=output.ptr();


       arm_add_f32(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksF32::vec_sub_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *inp2=input2.ptr();
       float32_t *outp=output.ptr();


       arm_sub_f32(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksF32::vec_abs_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *inp2=input2.ptr();
       float32_t *outp=output.ptr();


       arm_abs_f32(inp1,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksF32::vec_negate_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *outp=output.ptr();


       arm_negate_f32(inp1,outp,this->nb);
        
    }

    void BasicMathsBenchmarksF32::vec_offset_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *inp2=input2.ptr();
       float32_t *outp=output.ptr();


       arm_offset_f32(inp1,1.0,outp,this->nb);
        
    }

   void BasicMathsBenchmarksF32::vec_scale_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *inp2=input2.ptr();
       float32_t *outp=output.ptr();


       arm_scale_f32(inp1,1.0,outp,this->nb);
        
    }

    void BasicMathsBenchmarksF32::vec_dot_f32()
    {
       
       float32_t *inp1=input1.ptr();
       float32_t *inp2=input2.ptr();
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
       
    }

    void BasicMathsBenchmarksF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
