#include "ComplexMathsBenchmarksQ15.h"
#include "Error.h"

   
    void ComplexMathsBenchmarksQ15::vec_conj_q15()
    {
       
       const q15_t *inp1=input1.ptr();
       q15_t *outp=output.ptr();


       arm_cmplx_conj_q15(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksQ15::vec_dot_prod_q15()
    {
       
       const q15_t *inp1=input1.ptr();
       const q15_t *inp2=input2.ptr();
       q31_t real,imag;


       arm_cmplx_dot_prod_q15(inp1,inp2,this->nb,&real,&imag);
        
    } 

    void ComplexMathsBenchmarksQ15::vec_mag_q15()
    {
       
       const q15_t *inp1=input1.ptr();
       q15_t *outp=output.ptr();


       arm_cmplx_mag_q15(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksQ15::vec_mag_squared_q15()
    {
       
       const q15_t *inp1=input1.ptr();
       q15_t *outp=output.ptr();


       arm_cmplx_mag_squared_q15(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksQ15::vec_mult_cmplx_q15()
    {
       
       const q15_t *inp1=input1.ptr();
       const q15_t *inp2=input2.ptr();
       q15_t *outp=output.ptr();


       arm_cmplx_mult_cmplx_q15(inp1,inp2,outp,this->nb);
        
    }

    void ComplexMathsBenchmarksQ15::vec_mult_real_q15()
    {
       
       const q15_t *inp1=input1.ptr();
       // Real input
       const q15_t *inp3=input3.ptr();
       q15_t *outp=output.ptr();


       arm_cmplx_mult_real_q15(inp1,inp3,outp,this->nb);
        
    }

   
    void ComplexMathsBenchmarksQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(ComplexMathsBenchmarksQ15::INPUT1_Q15_ID,mgr,this->nb);
       input2.reload(ComplexMathsBenchmarksQ15::INPUT2_Q15_ID,mgr,this->nb);
       input3.reload(ComplexMathsBenchmarksQ15::INPUT3_Q15_ID,mgr,this->nb);
       
       output.create(this->nb,ComplexMathsBenchmarksQ15::OUT_SAMPLES_Q15_ID,mgr);
       
    }

    void ComplexMathsBenchmarksQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
