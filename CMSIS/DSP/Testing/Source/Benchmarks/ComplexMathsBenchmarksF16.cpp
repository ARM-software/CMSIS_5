#include "ComplexMathsBenchmarksF16.h"
#include "Error.h"

   
    void ComplexMathsBenchmarksF16::vec_conj_f16()
    {
       arm_cmplx_conj_f16(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksF16::vec_dot_prod_f16()
    {
       float16_t real,imag;
       arm_cmplx_dot_prod_f16(this->inp1,this->inp2,this->nb,&real,&imag);
    } 

    void ComplexMathsBenchmarksF16::vec_mag_f16()
    {
       arm_cmplx_mag_f16(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksF16::vec_mag_squared_f16()
    {
       arm_cmplx_mag_squared_f16(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksF16::vec_mult_cmplx_f16()
    {
      arm_cmplx_mult_cmplx_f16(this->inp1,this->inp2,this->outp,this->nb);
    }

    void ComplexMathsBenchmarksF16::vec_mult_real_f16()
    {
       arm_cmplx_mult_real_f16(this->inp1,this->inp3,this->outp,this->nb);
    }

   
    void ComplexMathsBenchmarksF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(ComplexMathsBenchmarksF16::INPUT1_F16_ID,mgr,this->nb);
       input2.reload(ComplexMathsBenchmarksF16::INPUT2_F16_ID,mgr,this->nb);
       input3.reload(ComplexMathsBenchmarksF16::INPUT3_F16_ID,mgr,this->nb);
       
       output.create(this->nb,ComplexMathsBenchmarksF16::OUT_SAMPLES_F16_ID,mgr);


       switch(id){
         case ComplexMathsBenchmarksF16::VEC_CONJ_F16_1:
         case ComplexMathsBenchmarksF16::VEC_MAG_F16_3:
         case ComplexMathsBenchmarksF16::VEC_MAG_SQUARED_F16_4:
             this->inp1=input1.ptr();
             this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksF16::VEC_DOT_PROD_F16_2:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
         break;

         case ComplexMathsBenchmarksF16::VEC_MULT_CMPLX_F16_5:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
            this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksF16::VEC_MULT_REAL_F16_6:
            this->inp1=input1.ptr();
            // Real input
            this->inp3=input3.ptr();
            this->outp=output.ptr();
         break;
       }
       
    }

    void ComplexMathsBenchmarksF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
    }
