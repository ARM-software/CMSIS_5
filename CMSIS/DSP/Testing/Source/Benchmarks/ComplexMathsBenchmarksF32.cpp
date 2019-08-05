#include "ComplexMathsBenchmarksF32.h"
#include "Error.h"

   
    void ComplexMathsBenchmarksF32::vec_conj_f32()
    {
       
       const float32_t *inp1=input1.ptr();
       float32_t *outp=output.ptr();


       arm_cmplx_conj_f32(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksF32::vec_dot_prod_f32()
    {
       
       const float32_t *inp1=input1.ptr();
       const float32_t *inp2=input2.ptr();
       float32_t real,imag;


       arm_cmplx_dot_prod_f32(inp1,inp2,this->nb,&real,&imag);
        
    } 

    void ComplexMathsBenchmarksF32::vec_mag_f32()
    {
       
       const float32_t *inp1=input1.ptr();
       float32_t *outp=output.ptr();


       arm_cmplx_mag_f32(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksF32::vec_mag_squared_f32()
    {
       
       const float32_t *inp1=input1.ptr();
       float32_t *outp=output.ptr();


       arm_cmplx_mag_squared_f32(inp1,outp,this->nb);
        
    } 

    void ComplexMathsBenchmarksF32::vec_mult_cmplx_f32()
    {
       
       const float32_t *inp1=input1.ptr();
       const float32_t *inp2=input2.ptr();
       float32_t *outp=output.ptr();


       arm_cmplx_mult_cmplx_f32(inp1,inp2,outp,this->nb);
        
    }

    void ComplexMathsBenchmarksF32::vec_mult_real_f32()
    {
       
       const float32_t *inp1=input1.ptr();
       // Real input
       const float32_t *inp3=input3.ptr();
       float32_t *outp=output.ptr();


       arm_cmplx_mult_real_f32(inp1,inp3,outp,this->nb);
        
    }

   
    void ComplexMathsBenchmarksF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(ComplexMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb);
       input2.reload(ComplexMathsBenchmarksF32::INPUT2_F32_ID,mgr,this->nb);
       input3.reload(ComplexMathsBenchmarksF32::INPUT3_F32_ID,mgr,this->nb);
       
       output.create(this->nb,ComplexMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);
       
    }

    void ComplexMathsBenchmarksF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
