#include "ComplexMathsBenchmarksF32.h"
#include "Error.h"

   
    void ComplexMathsBenchmarksF32::vec_conj_f32()
    {
       arm_cmplx_conj_f32(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksF32::vec_dot_prod_f32()
    {
       float32_t real,imag;
       arm_cmplx_dot_prod_f32(this->inp1,this->inp2,this->nb,&real,&imag);
    } 

    void ComplexMathsBenchmarksF32::vec_mag_f32()
    {
       arm_cmplx_mag_f32(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksF32::vec_mag_squared_f32()
    {
       arm_cmplx_mag_squared_f32(this->inp1,this->outp,this->nb);
    } 

    void ComplexMathsBenchmarksF32::vec_mult_cmplx_f32()
    {
      arm_cmplx_mult_cmplx_f32(this->inp1,this->inp2,this->outp,this->nb);
    }

    void ComplexMathsBenchmarksF32::vec_mult_real_f32()
    {
       arm_cmplx_mult_real_f32(this->inp1,this->inp3,this->outp,this->nb);
    }

   
    void ComplexMathsBenchmarksF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

       input1.reload(ComplexMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb);
       input2.reload(ComplexMathsBenchmarksF32::INPUT2_F32_ID,mgr,this->nb);
       input3.reload(ComplexMathsBenchmarksF32::INPUT3_F32_ID,mgr,this->nb);
       
       output.create(this->nb,ComplexMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);


       switch(id){
         case ComplexMathsBenchmarksF32::VEC_CONJ_F32_1:
         case ComplexMathsBenchmarksF32::VEC_MAG_F32_3:
         case ComplexMathsBenchmarksF32::VEC_MAG_SQUARED_F32_4:
             this->inp1=input1.ptr();
             this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksF32::VEC_DOT_PROD_F32_2:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
         break;

         case ComplexMathsBenchmarksF32::VEC_MULT_CMPLX_F32_5:
            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
            this->outp=output.ptr();
         break;

         case ComplexMathsBenchmarksF32::VEC_MULT_REAL_F32_6:
            this->inp1=input1.ptr();
            // Real input
            this->inp3=input3.ptr();
            this->outp=output.ptr();
         break;
       }
       
    }

    void ComplexMathsBenchmarksF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
