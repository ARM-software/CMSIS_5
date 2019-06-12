#include "BasicMathsBenchmarksQ31.h"
#include "Error.h"

   
    void BasicMathsBenchmarksQ31::vec_mult_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();


       arm_mult_q31(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ31::vec_add_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();


       arm_add_q31(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ31::vec_sub_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();


       arm_sub_q31(inp1,inp2,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ31::vec_abs_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();


       arm_abs_q31(inp1,outp,this->nb);
        
    } 

    void BasicMathsBenchmarksQ31::vec_negate_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *outp=output.ptr();


       arm_negate_q31(inp1,outp,this->nb);
        
    }

    void BasicMathsBenchmarksQ31::vec_offset_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();


       arm_offset_q31(inp1,1.0,outp,this->nb);
        
    }

   void BasicMathsBenchmarksQ31::vec_scale_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();


       arm_scale_q31(inp1,0x45,1,outp,this->nb);
        
    }

    void BasicMathsBenchmarksQ31::vec_dot_q31()
    {
       
       q31_t *inp1=input1.ptr();
       q31_t *inp2=input2.ptr();
       q63_t result;


       arm_dot_prod_q31(inp1,inp2,this->nb,&result);
        
    }

  
    
    void BasicMathsBenchmarksQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(BasicMathsBenchmarksQ31::INPUT1_Q31_ID,mgr,this->nb);
       input2.reload(BasicMathsBenchmarksQ31::INPUT2_Q31_ID,mgr,this->nb);

       
       output.create(this->nb,BasicMathsBenchmarksQ31::OUT_SAMPLES_Q31_ID,mgr);
       
    }

    void BasicMathsBenchmarksQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
