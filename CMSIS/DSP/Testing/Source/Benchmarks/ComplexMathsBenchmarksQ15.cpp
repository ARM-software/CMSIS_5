#include "ComplexMathsBenchmarksQ15.h"
#include "Error.h"

   
    void ComplexMathsBenchmarksQ15::vec_conj_q15()
    {
       arm_cmplx_conj_q15(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksQ15::vec_dot_prod_q15()
    {
       q31_t real,imag;

       arm_cmplx_dot_prod_q15(this->inp1,this->inp2,this->nb,&real,&imag);
    } 

    void ComplexMathsBenchmarksQ15::vec_mag_q15()
    {
       arm_cmplx_mag_q15(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksQ15::vec_mag_squared_q15()
    {
       arm_cmplx_mag_squared_q15(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksQ15::vec_mult_cmplx_q15()
    {
       arm_cmplx_mult_cmplx_q15(this->inp1,this->inp2,this->outp,this->nb);
    }

    void ComplexMathsBenchmarksQ15::vec_mult_real_q15()
    {
       arm_cmplx_mult_real_q15(this->inp1,this->inp3,this->outp,this->nb);
    }

   
    void ComplexMathsBenchmarksQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(ComplexMathsBenchmarksQ15::INPUT1_Q15_ID,mgr,this->nb);
       input2.reload(ComplexMathsBenchmarksQ15::INPUT2_Q15_ID,mgr,this->nb);
       input3.reload(ComplexMathsBenchmarksQ15::INPUT3_Q15_ID,mgr,this->nb);
       
       output.create(this->nb,ComplexMathsBenchmarksQ15::OUT_SAMPLES_Q15_ID,mgr);

       switch(id){
         case ComplexMathsBenchmarksQ15::VEC_CONJ_Q15_1:
         case ComplexMathsBenchmarksQ15::VEC_MAG_Q15_3:
         case ComplexMathsBenchmarksQ15::VEC_MAG_SQUARED_Q15_4:
             this->inp1=input1.ptr();
             this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksQ15::VEC_DOT_PROD_Q15_2:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
         break;

         case ComplexMathsBenchmarksQ15::VEC_MULT_CMPLX_Q15_5:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
            this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksQ15::VEC_MULT_REAL_Q15_6:
            this->inp1=input1.ptr();
            // Real input
            this->inp3=input3.ptr();
            this->outp=output.ptr();
         break;
       }
       
    }

    void ComplexMathsBenchmarksQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
