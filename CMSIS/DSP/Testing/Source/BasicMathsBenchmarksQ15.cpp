#include "BasicMathsBenchmarksQ15.h"
#include "Error.h"

   
    void BasicMathsBenchmarksQ15::vec_mult_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *inp2=input2.ptr();
       q15_t *outp=output.ptr();


       arm_mult_q15(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ15::vec_add_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *inp2=input2.ptr();
       q15_t *outp=output.ptr();


       arm_add_q15(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ15::vec_sub_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *inp2=input2.ptr();
       q15_t *outp=output.ptr();


       arm_sub_q15(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ15::vec_abs_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *inp2=input2.ptr();
       q15_t *outp=output.ptr();


       arm_abs_q15(inp1,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ15::vec_negate_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *outp=output.ptr();


       arm_negate_q15(inp1,outp,this->nb);
        
    }

    void BasicMathsBenchmarksQ15::vec_offset_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *inp2=input2.ptr();
       q15_t *outp=output.ptr();


       arm_offset_q15(inp1,1.0,outp,this->nb);
        
    }

   void BasicMathsBenchmarksQ15::vec_scale_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *inp2=input2.ptr();
       q15_t *outp=output.ptr();


       arm_scale_q15(inp1,0x45,1,outp,this->nb);
        
    }

    void BasicMathsBenchmarksQ15::vec_dot_q15()
    {
       
       q15_t *inp1=input1.ptr();
       q15_t *inp2=input2.ptr();
       q63_t result;


       arm_dot_prod_q15(inp1,inp2,this->nb,&result);
        
    }

  
    
    void BasicMathsBenchmarksQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksQ15::INPUT1_Q15_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksQ15::INPUT2_Q15_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksQ15::OUT_SAMPLES_Q15_ID,mgr);
       
    }

    void BasicMathsBenchmarksQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
