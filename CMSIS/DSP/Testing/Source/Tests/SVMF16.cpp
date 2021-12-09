#include "SVMF16.h"
#include <stdio.h>
#include "Error.h"


void SVMF16::test_svm_linear_predict_f16()
{
      const float16_t *inp  = samples.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_linear_predict_f16(&this->linear,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 


void SVMF16::test_svm_polynomial_predict_f16()
{
      const float16_t *inp  = samples.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_polynomial_predict_f16(&this->poly,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 

void SVMF16::test_svm_rbf_predict_f16()
{
      const float16_t *inp  = samples.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_rbf_predict_f16(&this->rbf,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 

void SVMF16::test_svm_sigmoid_predict_f16()
{
      const float16_t *inp  = samples.ptr();
      int32_t *outp         = output.ptr();
      int32_t *result;

      result=outp;

      for(int i =0; i < this->nbTestSamples; i++)
      {
         arm_svm_sigmoid_predict_f16(&this->sigmoid,inp,result);
         result++;
         inp += this->vecDim;
      }

      ASSERT_EQ(ref,output);

} 



void SVMF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& testparams,Client::PatternMgr *mgr)
{
      
      int kind;
      Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
      (void)testparams;

      switch(id)
      {
          case SVMF16::TEST_SVM_LINEAR_PREDICT_F16_1:
          {
             
             samples.reload(SVMF16::SAMPLES1_F16_ID,mgr,nb);
             params.reload(SVMF16::PARAMS1_F16_ID,mgr,nb);
             dims.reload(SVMF16::DIMS1_S16_ID,mgr,nb);
             ref.reload(SVMF16::REF1_S32_ID,mgr,nb);
          }
          break;

          case SVMF16::TEST_SVM_POLYNOMIAL_PREDICT_F16_2:
          {
             
             samples.reload(SVMF16::SAMPLES2_F16_ID,mgr,nb);
             params.reload(SVMF16::PARAMS2_F16_ID,mgr,nb);
             dims.reload(SVMF16::DIMS2_S16_ID,mgr,nb);
             ref.reload(SVMF16::REF2_S32_ID,mgr,nb);
          }
          break;

          case SVMF16::TEST_SVM_RBF_PREDICT_F16_3:
          {
             
             samples.reload(SVMF16::SAMPLES3_F16_ID,mgr,nb);
             params.reload(SVMF16::PARAMS3_F16_ID,mgr,nb);
             dims.reload(SVMF16::DIMS3_S16_ID,mgr,nb);
             ref.reload(SVMF16::REF3_S32_ID,mgr,nb);
          }
          break;

          case SVMF16::TEST_SVM_SIGMOID_PREDICT_F16_4:
          {
             
             samples.reload(SVMF16::SAMPLES4_F16_ID,mgr,nb);
             params.reload(SVMF16::PARAMS4_F16_ID,mgr,nb);
             dims.reload(SVMF16::DIMS4_S16_ID,mgr,nb);
             ref.reload(SVMF16::REF4_S32_ID,mgr,nb);
          }
          break;
#if 0
          case SVMF16::TEST_SVM_RBF_PREDICT_F16_5:
          {
             
             samples.reload(SVMF16::SAMPLES5_F16_ID,mgr,nb);
             params.reload(SVMF16::PARAMS5_F16_ID,mgr,nb);
             dims.reload(SVMF16::DIMS5_S16_ID,mgr,nb);
             ref.reload(SVMF16::REF5_S32_ID,mgr,nb);
          }
          break;
#endif
      }


      
      
      const int16_t   *dimsp = dims.ptr();
      const float16_t  *paramsp = params.ptr();
      
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

        
         case SVMF16::POLY:
             this->degree = dimsp[6];
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
         break;

         case SVMF16::RBF:
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1];
         break;

         case SVMF16::SIGMOID:
             this->coef0 =paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 1] ;
             this->gamma=paramsp[this->vecDim*this->nbSupportVectors + this->nbSupportVectors + 2];
         break;
      }

       
       switch(id)
       {
          case SVMF16::TEST_SVM_LINEAR_PREDICT_F16_1:
          //case SVMF16::TEST_SVM_RBF_PREDICT_F16_5:
          {
             
             arm_svm_linear_init_f16(&linear, 
                 this->nbSupportVectors,
                 this->vecDim,
                 this->intercept,
                 this->dualCoefs,
                 this->supportVectors,
                 this->classes);
          }
          break;

          case SVMF16::TEST_SVM_POLYNOMIAL_PREDICT_F16_2:
          {
             
             arm_svm_polynomial_init_f16(&poly, 
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

          case SVMF16::TEST_SVM_RBF_PREDICT_F16_3:
          {
             
             arm_svm_rbf_init_f16(&rbf, 
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

          case SVMF16::TEST_SVM_SIGMOID_PREDICT_F16_4:
          {
             
             arm_svm_sigmoid_init_f16(&sigmoid, 
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


       output.create(ref.nbSamples(),SVMF16::OUT_S32_ID,mgr);
    
}

void SVMF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
{
        (void)id;
        output.dump(mgr);
}



