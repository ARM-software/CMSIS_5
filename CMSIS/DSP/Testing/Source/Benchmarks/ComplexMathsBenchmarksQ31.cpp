#include "ComplexMathsBenchmarksQ31.h"
#include "Error.h"

   
    void ComplexMathsBenchmarksQ31::vec_conj_q31()
    {
       arm_cmplx_conj_q31(this->inp1,this->outp,this->nb); 
    } 

    void ComplexMathsBenchmarksQ31::vec_dot_prod_q31()
    {
       q63_t real,imag;

       arm_cmplx_dot_prod_q31(this->inp1,this->inp2,this->nb,&real,&imag);
    } 

    void ComplexMathsBenchmarksQ31::vec_mag_q31()
    {
       arm_cmplx_mag_q31(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksQ31::vec_mag_squared_q31()
    {
       arm_cmplx_mag_squared_q31(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksQ31::vec_mult_cmplx_q31()
    {
       arm_cmplx_mult_cmplx_q31(this->inp1,this->inp2,this->outp,this->nb);
    }

    void ComplexMathsBenchmarksQ31::vec_mult_real_q31()
    {
       arm_cmplx_mult_real_q31(this->inp1,this->inp3,this->outp,this->nb);
    }

   
    void ComplexMathsBenchmarksQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(ComplexMathsBenchmarksQ31::INPUT1_Q31_ID,mgr,this->nb);
       input2.reload(ComplexMathsBenchmarksQ31::INPUT2_Q31_ID,mgr,this->nb);
       input3.reload(ComplexMathsBenchmarksQ31::INPUT3_Q31_ID,mgr,this->nb);
       
       output.create(this->nb,ComplexMathsBenchmarksQ31::OUT_SAMPLES_Q31_ID,mgr);
       
       switch(id){
         case ComplexMathsBenchmarksQ31::VEC_CONJ_Q31_1:
         case ComplexMathsBenchmarksQ31::VEC_MAG_Q31_3:
         case ComplexMathsBenchmarksQ31::VEC_MAG_SQUARED_Q31_4:
             this->inp1=input1.ptr();
             this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksQ31::VEC_DOT_PROD_Q31_2:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
         break;

         case ComplexMathsBenchmarksQ31::VEC_MULT_CMPLX_Q31_5:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
            this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksQ31::VEC_MULT_REAL_Q31_6:
            this->inp1=input1.ptr();
            // Real input
            this->inp3=input3.ptr();
            this->outp=output.ptr();
         break;
       }
    }

    void ComplexMathsBenchmarksQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
