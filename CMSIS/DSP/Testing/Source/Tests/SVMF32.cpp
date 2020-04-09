#include "SVMF32.h"
#include <stdio.h>
#include "Error.h"


void SVMF32::test_svm_linear_predict_f32()
{
      const float32_t *inp  = samples.ptr();
      int32_t *refp         = ref.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_linear_predict_f32(&this->linear,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 


void SVMF32::test_svm_polynomial_predict_f32()
{
      const float32_t *inp  = samples.ptr();
      int32_t *refp         = ref.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_polynomial_predict_f32(&this->poly,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 

void SVMF32::test_svm_rbf_predict_f32()
{
      const float32_t *inp  = samples.ptr();
      int32_t *refp         = ref.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_rbf_predict_f32(&this->rbf,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 

void SVMF32::test_svm_sigmoid_predict_f32()
{
      const float32_t *inp  = samples.ptr();
      int32_t *refp         = ref.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_sigmoid_predict_f32(&this->sigmoid,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 

void SVMF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& testparams,Client::PatternMgr *mgr)
{
      
      int kind;
      Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

      switch(id)
      {
          case SVMF32::TEST_SVM_LINEAR_PREDICT_F32_1:
          {
             
             samples.reload(SVMF32::SAMPLES1_F32_ID,mgr,nb);
             params.reload(SVMF32::PARAMS1_F32_ID,mgr,nb);
             dims.reload(SVMF32::DIMS1_S16_ID,mgr,nb);
             ref.reload(SVMF32::REF1_S32_ID,mgr,nb);
          }
          break;

          case SVMF32::TEST_SVM_POLYNOMIAL_PREDICT_F32_2:
          {
             
             samples.reload(SVMF32::SAMPLES2_F32_ID,mgr,nb);
             params.reload(SVMF32::PARAMS2_F32_ID,mgr,nb);
             dims.reload(SVMF32::DIMS2_S16_ID,mgr,nb);
             ref.reload(SVMF32::REF2_S32_ID,mgr,nb);
          }
          break;

          case SVMF32::TEST_SVM_RBF_PREDICT_F32_3:
          {
             
             samples.reload(SVMF32::SAMPLES3_F32_ID,mgr,nb);
             params.reload(SVMF32::PARAMS3_F32_ID,mgr,nb);
             dims.reload(SVMF32::DIMS3_S16_ID,mgr,nb);
             ref.reload(SVMF32::REF3_S32_ID,mgr,nb);
          }
          break;

          case SVMF32::TEST_SVM_SIGMOID_PREDICT_F32_4:
          {
             
             samples.reload(SVMF32::SAMPLES4_F32_ID,mgr,nb);
             params.reload(SVMF32::PARAMS4_F32_ID,mgr,nb);
             dims.reload(SVMF32::DIMS4_S16_ID,mgr,nb);
             ref.reload(SVMF32::REF4_S32_ID,mgr,nb);
          }
          break;

          case SVMF32::TEST_SVM_RBF_PREDICT_F32_5:
          {
             
             samples.reload(SVMF32::SAMPLES5_F32_ID,mgr,nb);
             params.reload(SVMF32::PARAMS5_F32_ID,mgr,nb);
             dims.reload(SVMF32::DIMS5_S16_ID,mgr,nb);
             ref.reload(SVMF32::REF5_S32_ID,mgr,nb);
          }
          break;
      }


      
      
      const int16_t   *dimsp = dims.ptr();
      const float32_t  *paramsp = params.ptr();
      
      kind = dimsp[0];
      
      this->classes[0] = dimsp[1];
      this->classes[1] = dimsp[2];
      this->nbTestSamples=dimsp[3];
      this->vecDim = dimsp[4];
      this->nbSupportVectors = dimsp[5];
      this->intercept=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors];
      this->supportVectors=paramsp;
      this->dualCoefs=paramsp + (this->vecDim*this->nbSupportVectors);

      switch(kind)
      {

        
         case SVMF32::POLY:
             this->degree = dimsp[6];
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
         break;

         case SVMF32::RBF:
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1];
         break;

         case SVMF32::SIGMOID:
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
         break;
      }

       
       switch(id)
       {
          case SVMF32::TEST_SVM_LINEAR_PREDICT_F32_1:
          {
             
             arm_svm_linear_init_f32(&linear, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes);
          }
          break;

          case SVMF32::TEST_SVM_POLYNOMIAL_PREDICT_F32_2:
          {
             
             arm_svm_polynomial_init_f32(&poly, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes,
                 this->degree,
                 this->coef0,
                 this->gamma
                 );
          }
          break;

          case SVMF32::TEST_SVM_RBF_PREDICT_F32_3:
          case SVMF32::TEST_SVM_RBF_PREDICT_F32_5:
          {
             
             arm_svm_rbf_init_f32(&rbf, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes,
                 this->gamma
                 );
          }
          break;

          case SVMF32::TEST_SVM_SIGMOID_PREDICT_F32_4:
          {
             
             arm_svm_sigmoid_init_f32(&sigmoid, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes,
                 this->coef0,
                 this->gamma
                 );
          }
          break;
       }


       output.create(ref.nbSamples(),SVMF32::OUT_S32_ID,mgr);
    
}

void SVMF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
{
        output.dump(mgr);
}



