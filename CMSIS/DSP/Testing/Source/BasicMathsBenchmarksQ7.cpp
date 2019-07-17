#include "BasicMathsBenchmarksQ7.h"
#include "Error.h"

   
    void BasicMathsBenchmarksQ7::vec_mult_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *inp2=input2.ptr();
       q7_t *outp=output.ptr();


       arm_mult_q7(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ7::vec_add_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *inp2=input2.ptr();
       q7_t *outp=output.ptr();


       arm_add_q7(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ7::vec_sub_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *inp2=input2.ptr();
       q7_t *outp=output.ptr();


       arm_sub_q7(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ7::vec_abs_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *inp2=input2.ptr();
       q7_t *outp=output.ptr();


       arm_abs_q7(inp1,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ7::vec_negate_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *outp=output.ptr();


       arm_negate_q7(inp1,outp,this->nb);
        
    }

    void BasicMathsBenchmarksQ7::vec_offset_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *inp2=input2.ptr();
       q7_t *outp=output.ptr();


       arm_offset_q7(inp1,1.0,outp,this->nb);
        
    }

   void BasicMathsBenchmarksQ7::vec_scale_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *inp2=input2.ptr();
       q7_t *outp=output.ptr();


       arm_scale_q7(inp1,0x45,1,outp,this->nb);
        
    }

    void BasicMathsBenchmarksQ7::vec_dot_q7()
    {
       
       q7_t *inp1=input1.ptr();
       q7_t *inp2=input2.ptr();
       q31_t result;


       arm_dot_prod_q7(inp1,inp2,this->nb,&result);
        
    }

  
    
    void BasicMathsBenchmarksQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksQ7::INPUT1_Q7_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksQ7::INPUT2_Q7_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksQ7::OUT_SAMPLES_Q7_ID,mgr);
       
    }

    void BasicMathsBenchmarksQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
