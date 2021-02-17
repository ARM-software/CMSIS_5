#include "QuaternionMathsBenchmarksF32.h"
#include "Error.h"

   
    void QuaternionMathsBenchmarksF32::test_quaternion_norm_f32()
    {
        arm_quaternion_norm_f32(this->inp1,this->outp,this->nb);
        
    } 

    void QuaternionMathsBenchmarksF32::test_quaternion_inverse_f32()
    {
       
        arm_quaternion_inverse_f32(this->inp1,this->outp,this->nb);

        
    } 

    void QuaternionMathsBenchmarksF32::test_quaternion_conjugate_f32()
    {
        
        arm_quaternion_conjugate_f32(this->inp1,this->outp,this->nb);

    } 

    void QuaternionMathsBenchmarksF32::test_quaternion_normalize_f32()
    {
        
        arm_quaternion_normalize_f32(this->inp1,this->outp,this->nb);

       
    } 

    void QuaternionMathsBenchmarksF32::test_quaternion_prod_single_f32()
    {
       
        for(int i=0; i < this->nb; i++)
        {
           arm_quaternion_product_single_f32(this->inp1,this->inp2,this->outp);
           this->outp += 4;
           this->inp1 += 4;
           this->inp2 += 4;
        }

    } 

    void QuaternionMathsBenchmarksF32::test_quaternion_product_f32()
    {
        
        arm_quaternion_product_f32(this->inp1,this->inp2,outp,this->nb);

       

    } 

    void QuaternionMathsBenchmarksF32::test_quaternion2rotation_f32()
    {
        arm_quaternion2rotation_f32(this->inp1,this->outp,this->nb);

      

    } 

    void QuaternionMathsBenchmarksF32::test_rotation2quaternion_f32()
    {
       

        arm_rotation2quaternion_f32(this->inp1,this->outp,this->nb);

        

    } 

    
    void QuaternionMathsBenchmarksF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {

       this->setForceInCache(true);
       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nb = *it;

      
       switch(id)
       {
          case QuaternionMathsBenchmarksF32::TEST_QUATERNION_NORM_F32_1:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb*4);
            output.create(this->nb,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->outp=output.ptr();
          break;

          case QuaternionMathsBenchmarksF32::TEST_QUATERNION_INVERSE_F32_2:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb*4);
            output.create(this->nb*4,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->outp=output.ptr();
          break;

          case QuaternionMathsBenchmarksF32::TEST_QUATERNION_CONJUGATE_F32_3:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb*4);
            output.create(this->nb*4,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->outp=output.ptr();
          break;

          case QuaternionMathsBenchmarksF32::TEST_QUATERNION_NORMALIZE_F32_4:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb*4);
            output.create(this->nb*4,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->outp=output.ptr();
          break;

          case QuaternionMathsBenchmarksF32::TEST_QUATERNION_PROD_SINGLE_F32_5:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb*4);
            input2.reload(QuaternionMathsBenchmarksF32::INPUT2_F32_ID,mgr,this->nb*4);
            output.create(this->nb*4,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
            this->outp=output.ptr();
          break;

          case QuaternionMathsBenchmarksF32::TEST_QUATERNION_PRODUCT_F32_6:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb*4);
            input2.reload(QuaternionMathsBenchmarksF32::INPUT2_F32_ID,mgr,this->nb*4);
            output.create(this->nb*4,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->inp2=input2.ptr();
            this->outp=output.ptr();
          break;

          case QuaternionMathsBenchmarksF32::TEST_QUATERNION2ROTATION_F32_7:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT1_F32_ID,mgr,this->nb*4);
            output.create(this->nb*9,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->outp=output.ptr();
          break;

          case QuaternionMathsBenchmarksF32::TEST_ROTATION2QUATERNION_F32_8:
            input1.reload(QuaternionMathsBenchmarksF32::INPUT_ROT_F32_ID,mgr,this->nb*9);
            output.create(this->nb*4,QuaternionMathsBenchmarksF32::OUT_SAMPLES_F32_ID,mgr);

            this->inp1=input1.ptr();
            this->outp=output.ptr();
          break;

       }
    }

    void QuaternionMathsBenchmarksF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
