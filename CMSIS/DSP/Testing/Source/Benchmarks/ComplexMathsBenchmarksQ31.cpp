#include "ComplexMathsBenchmarksQ31.h"
#include "Error.h"

   
    void ComplexMathsBenchmarksQ31::vec_conj_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       q31_t *outp=output.ptr();


       arm_cmplx_conj_q31(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksQ31::vec_dot_prod_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       const q31_t *inp2=input2.ptr();
       q63_t real,imag;


       arm_cmplx_dot_prod_q31(inp1,inp2,this->nb,&real,&imag);
        
    } 

    void ComplexMathsBenchmarksQ31::vec_mag_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       q31_t *outp=output.ptr();


       arm_cmplx_mag_q31(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksQ31::vec_mag_squared_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       q31_t *outp=output.ptr();


       arm_cmplx_mag_squared_q31(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksQ31::vec_mult_cmplx_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       const q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();


       arm_cmplx_mult_cmplx_q31(inp1,inp2,outp,this->nb);
        
    }

    void ComplexMathsBenchmarksQ31::vec_mult_real_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       // Real input
       const q31_t *inp3=input3.ptr();
       q31_t *outp=output.ptr();


       arm_cmplx_mult_real_q31(inp1,inp3,outp,this->nb);
        
    }

   
    void ComplexMathsBenchmarksQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(ComplexMathsBenchmarksQ31::INPUT1_Q31_ID,mgr,this->nb);
       input2.reload(ComplexMathsBenchmarksQ31::INPUT2_Q31_ID,mgr,this->nb);
       input3.reload(ComplexMathsBenchmarksQ31::INPUT3_Q31_ID,mgr,this->nb);
       
       output.create(this->nb,ComplexMathsBenchmarksQ31::OUT_SAMPLES_Q31_ID,mgr);
       
    }

    void ComplexMathsBenchmarksQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
